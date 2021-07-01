import logging
import pathlib
from math import pi

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as tfr
from tqdm import trange

import rospy
import urdf_parser_py.xml_reflection.core
from arc_utilities.ros_helpers import get_connected_publisher
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from moveit_msgs.msg import DisplayRobotState
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, kdl_chain_from_urdf_model
from tf_robot_learning.kinematic.utils.layout import FkLayout
from visualization_msgs.msg import Marker


def orientation_error_quat(q1, q2):
    # NOTE: I don't know of a correct & smooth matrix -> quaternion implementation
    # https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    return 1 - tf.square(tf.einsum('bi,bi->b', q1, q2))


def orientation_error_mat(r1, r2):
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    I = tf.expand_dims(tf.eye(3), 0)
    R_Rt = tf.matmul(r1, tf.transpose(r2, [0, 2, 1]))
    return tf.linalg.norm(I - R_Rt, ord='fro', axis=[-2, -1])


def orientation_error_mat2(r1, r2):
    # NOTE: not differentiable
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    R_Rt = tf.matmul(r1, tf.transpose(r2, [0, 2, 1]))
    log = tf.cast(tf.linalg.logm(tf.cast(R_Rt, tf.complex64)), tf.float32)
    return tf.linalg.norm(log, ord='fro', axis=[-2, -1])


@tf.function
def pose_loss(chain, q, target_pose, theta=0.9):
    xs = chain.xs(q, layout=FkLayout.xm)
    position = xs[-1, :3]
    orientation = tf.reshape(xs[-1, 3:], [-1, 3, 3])
    target_position = target_pose[:, :3]
    target_quat = target_pose[:, 3:]
    target_orientation = tfr.from_quaternion(target_quat)
    _orientation_error = orientation_error_mat(target_orientation, orientation)
    position_error = tf.reduce_sum(tf.square(position - target_position), axis=-1)

    return xs, position_error, _orientation_error


def target(x, y, z, roll, pitch, yaw):
    return tf.cast(tf.expand_dims(tf.concat([[x, y, z], quaternion_from_euler(roll, pitch, yaw)], 0), 0), tf.float32)


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
    pub2 = get_connected_publisher("point", Marker, queue_size=10)
    pub3 = get_connected_publisher("joint_states_viz", JointState, queue_size=10)
    tf2 = TF2Wrapper()

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error
    urdf = urdf_from_file(urdf_filename.as_posix())

    left = kdl_chain_from_urdf_model(urdf, tip='left_tool')
    right = kdl_chain_from_urdf_model(urdf, tip='right_tool')

    robot = DisplayRobotState()
    joint_names = list(set(left.actuated_joint_names() + right.actuated_joint_names()))
    n_joints = len(joint_names)
    left_joint_indices = [joint_names.index(jn) for jn in left.actuated_joint_names()]
    right_joint_indices = [joint_names.index(jn) for jn in right.actuated_joint_names()]

    # lr = tf.keras.optimizers.schedules.ExponentialDecay(0.5, 20, 0.9)
    # opt = tf.keras.optimizers.SGD(lr)
    opt = tf.keras.optimizers.Adam(0.01)
    left_target_pose = target(-0.3, 0.5, 0.3, 0, -pi / 2, -pi / 2)
    right_target_pose = target(0.3, 0.5, 0.3, -pi / 2, -pi / 2, 0)
    q = tf.Variable([0.0] * n_joints, dtype=tf.float32)
    variables = [q]

    debug_viz = False
    theta = 0.99

    converged = False
    for i in trange(1000):
        def _step():
            with tf.GradientTape(persistent=True) as tape:
                left_q = tf.gather(q, left_joint_indices)
                right_q = tf.gather(q, right_joint_indices)
                left_xs, left_pos_error, left_rot_error = pose_loss(left, left_q, left_target_pose)
                right_xs, right_pos_error, right_rot_error = pose_loss(right, right_q, right_target_pose)
                left_loss = theta * left_pos_error + (1 - theta) * left_rot_error
                right_loss = theta * right_pos_error + (1 - theta) * right_rot_error
                loss = tf.reduce_mean(left_loss + right_loss)

            if debug_viz:
                print(tf.linalg.norm(tape.gradient(left_pos_error, variables)[0].values).numpy(),
                      tf.linalg.norm(tape.gradient(left_rot_error, variables)[0].values).numpy(),
                      tf.linalg.norm(tape.gradient(right_pos_error, variables)[0].values).numpy(),
                      tf.linalg.norm(tape.gradient(right_rot_error, variables)[0].values).numpy())
                print("LOSS", loss.numpy())
            gradients = tape.gradient([loss], variables)
            opt.apply_gradients(grads_and_vars=zip(gradients, variables))
            return loss, (left_q, right_q, left_xs, right_xs)

        loss, debug = _step()
        left_q, right_q, left_xs, right_xs = debug

        if loss < 1e-6:
            converged = True
            break

        # VIZ
        def _viz():
            b = 0
            tf2.send_transform(left_target_pose[b, :3].numpy().tolist(),
                               left_target_pose[b, 3:].numpy().tolist(),
                               parent='world', child='left_target')
            tf2.send_transform(right_target_pose[b, :3].numpy().tolist(),
                               right_target_pose[b, 3:].numpy().tolist(),
                               parent='world', child='right_target')

            positions = tf.concat([left_xs[:, :3], right_xs[:, :3]], axis=0)

            robot_state_dict = {}
            for name, position in zip(left.actuated_joint_names(), left_q.numpy().tolist()):
                robot_state_dict[name] = position
            for name, position in zip(right.actuated_joint_names(), right_q.numpy().tolist()):
                robot_state_dict[name] = position

            robot.state.joint_state.name = robot_state_dict.keys()
            robot.state.joint_state.position = robot_state_dict.values()
            robot.state.joint_state.header.stamp = rospy.Time.now()
            pub.publish(robot)
            pub3.publish(robot.state.joint_state)

            msg = Marker()
            msg.header.frame_id = 'world'
            msg.header.stamp = rospy.Time.now()
            msg.id = 0
            msg.type = Marker.SPHERE_LIST
            msg.action = Marker.ADD
            msg.pose.orientation.w = 1
            scale = 0.01
            msg.scale.x = scale
            msg.scale.y = scale
            msg.scale.z = scale
            msg.color.r = 1
            msg.color.a = 1
            for position in positions:
                p = Point(x=position[0], y=position[1], z=position[2])
                msg.points.append(p)

            pub2.publish(msg)

        if debug_viz:
            _viz()


if __name__ == '__main__':
    main()
