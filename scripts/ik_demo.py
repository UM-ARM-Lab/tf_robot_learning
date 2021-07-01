import pathlib

import tensorflow as tf

import rospy
import urdf_parser_py.xml_reflection.core
from arc_utilities.ros_helpers import get_connected_publisher
from geometry_msgs.msg import Point
from moveit_msgs.msg import DisplayRobotState
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, kdl_chain_from_urdf_model
from tf_robot_learning.kinematic.utils.layout import FkLayout
from visualization_msgs.msg import Marker


def pose_loss(chain, q, target_pose):
    xs = chain.xs(q, layout=FkLayout.xm)
    position = xs[:, :3]
    orientation = tf.reshape(xs[:, 3:], [-1, 3, 3])
    # TODO: orientation error
    target_position = target_pose[:3]
    loss = tf.reduce_sum(tf.square(position[-1] - target_position))
    return xs, loss


def main():
    rospy.init_node("ik_demo")

    pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
    pub2 = get_connected_publisher("point", Marker, queue_size=10)

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

    opt = tf.keras.optimizers.SGD(1.0)
    left_target_pose = [0.0, 0.0, 0.0, 0, 0, 0]
    right_target_pose = [0.3, 0.3, 0.3, 0, 0, 0]
    q = tf.Variable([0.0] * n_joints, dtype=tf.float32)
    variables = [q]
    for _ in range(1000):
        # OPT
        with tf.GradientTape() as tape:
            left_q = tf.gather(q, left_joint_indices)
            right_q = tf.gather(q, right_joint_indices)
            left_xs, left_loss = pose_loss(left, left_q, left_target_pose)
            right_xs, right_loss = pose_loss(right, right_q, right_target_pose)
            loss = left_loss + right_loss
        print(loss)
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(grads_and_vars=zip(gradients, variables))

        if loss < 1e-6:
            break

        # VIZ
        def _viz():
            positions = tf.concat([left_xs[:, :3], right_xs[:, :3]], axis=0)

            robot_state_dict = {}
            for name, position in zip(left.actuated_joint_names(), left_q.numpy().tolist()):
                robot_state_dict[name] = position
            for name, position in zip(right.actuated_joint_names(), right_q.numpy().tolist()):
                robot_state_dict[name] = position

            robot.state.joint_state.name = robot_state_dict.keys()
            robot.state.joint_state.position = robot_state_dict.values()
            pub.publish(robot)

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

        _viz()


if __name__ == '__main__':
    main()
