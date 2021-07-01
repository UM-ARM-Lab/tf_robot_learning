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
from tf_robot_learning.kinematic.chain import Chain
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
def compute_pose_loss(chain, q, target_pose):
    xs = chain.xs(q, layout=FkLayout.xm)
    position = xs[:, -1, :3]
    orientation = tf.reshape(xs[:, -1, 3:], [-1, 3, 3])
    target_position = target_pose[:, :3]
    target_quat = target_pose[:, 3:]
    target_orientation = tfr.from_quaternion(target_quat)
    _orientation_error = orientation_error_mat(target_orientation, orientation)
    position_error = tf.reduce_sum(tf.square(position - target_position), axis=-1)

    return xs, position_error, _orientation_error


def target(x, y, z, roll, pitch, yaw):
    return tf.cast(tf.expand_dims(tf.concat([[x, y, z], quaternion_from_euler(roll, pitch, yaw)], 0), 0), tf.float32)


class HdtIK:

    def __init__(self, urdf_filename: pathlib.Path):
        self.urdf = urdf_from_file(urdf_filename.as_posix())

        self.left = kdl_chain_from_urdf_model(self.urdf, tip='left_tool')
        self.right = kdl_chain_from_urdf_model(self.urdf, tip='right_tool')

        self.joint_names = list(set(self.left.actuated_joint_names() + self.right.actuated_joint_names()))
        self.n_joints = len(self.joint_names)
        self.left_idx = [self.joint_names.index(jn) for jn in self.left.actuated_joint_names()]
        self.right_idx = [self.joint_names.index(jn) for jn in self.right.actuated_joint_names()]

        self.theta = 0.99
        self.jl_alpha = 1.0
        self.initial_lr = 0.01

        # lr = tf.keras.optimizers.schedules.ExponentialDecay(0.5, 20, 0.9)
        # opt = tf.keras.optimizers.SGD(lr)
        self.optimizer = tf.keras.optimizers.Adam(self.initial_lr, amsgrad=True)

        self.display_robot_state_pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
        self.point_pub = get_connected_publisher("point", Marker, queue_size=10)
        self.joint_states_viz_pub = get_connected_publisher("joint_states_viz", JointState, queue_size=10)
        self.tf2 = TF2Wrapper()

    def solve(self, left_target_pose, right_target_pose, initial_value=None, viz=False):
        if initial_value is None:
            batch_size = left_target_pose.shape[0]
            initial_value = tf.zeros([batch_size, self.n_joints], dtype=tf.float32)
        q = tf.Variable(initial_value)

        converged = False
        for _ in trange(5000):
            loss, gradients, viz_info = self.opt(q, left_target_pose, right_target_pose)
            if loss < 5e-5:
                converged = True
                break

            if viz:
                self.viz_func(left_target_pose, right_target_pose, viz_info)

        return q, converged

    def opt(self, q, left_target_pose, right_target_pose):
        with tf.GradientTape(persistent=True) as tape:
            loss, viz_info = self.step(q, left_target_pose, right_target_pose)
        gradients = tape.gradient([loss], [q])
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, [q]))
        return loss, gradients, viz_info

    def step(self, q, left_target_pose, right_target_pose):
        left_q = tf.gather(q, self.left_idx, axis=1)
        left_pose_loss, left_xs = self.compute_pose_loss(self.left, left_q, left_target_pose)
        left_jl_loss = self.compute_jl_loss(self.left, left_q)
        right_q = tf.gather(q, self.right_idx, axis=1)
        right_pose_loss, right_xs = self.compute_pose_loss(self.right, right_q, right_target_pose)
        right_jl_loss = self.compute_jl_loss(self.right, right_q)
        loss = tf.reduce_mean(tf.math.add_n([left_pose_loss, right_pose_loss, left_jl_loss, right_jl_loss]))

        viz_info = [left_xs, right_xs, left_q, right_q]

        return loss, viz_info

    def compute_jl_loss(self, chain: Chain, q):
        joint_limits = chain.joint_limits
        jl_low = joint_limits[:, 0][tf.newaxis]
        jl_high = joint_limits[:, 1][tf.newaxis]
        low_error = tf.math.maximum(jl_low - q, 0)
        high_error = tf.math.maximum(q - jl_high, 0)
        jl_errors = tf.math.maximum(low_error, high_error)
        jl_loss = tf.reduce_sum(jl_errors, axis=-1)
        return self.jl_alpha * jl_loss

    def compute_pose_loss(self, chain: Chain, q, target_pose):
        xs, pos_error, rot_error = compute_pose_loss(chain, q, target_pose)
        pose_loss = self.theta * pos_error + (1 - self.theta) * rot_error
        return pose_loss, xs

    def viz_func(self, left_target_pose, right_target_pose, viz_info):
        left_xs, right_xs, left_q, right_q = viz_info
        b = 0
        self.tf2.send_transform(left_target_pose[b, :3].numpy().tolist(),
                                left_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='left_target')
        self.tf2.send_transform(right_target_pose[b, :3].numpy().tolist(),
                                right_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='right_target')

        positions = tf.concat([left_xs[b, :, :3], right_xs[b, :, :3]], axis=0)

        robot_state_dict = {}
        for name, position in zip(self.left.actuated_joint_names(), left_q[b].numpy().tolist()):
            robot_state_dict[name] = position
        for name, position in zip(self.right.actuated_joint_names(), right_q[b].numpy().tolist()):
            robot_state_dict[name] = position

        robot = DisplayRobotState()
        robot.state.joint_state.name = robot_state_dict.keys()
        robot.state.joint_state.position = robot_state_dict.values()
        robot.state.joint_state.header.stamp = rospy.Time.now()
        self.display_robot_state_pub.publish(robot)
        self.joint_states_viz_pub.publish(robot.state.joint_state)

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

        self.point_pub.publish(msg)

    def get_joint_names(self):
        return self.joint_names


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    ik_solver = HdtIK(urdf_filename)

    left_target_pose = target(-0.3, 0.5, 0.3, 0, -pi / 2, -pi / 2)
    right_target_pose = target(0.3, 0.5, 0.3, -pi / 2, -pi / 2, 0)

    q, converged = ik_solver.solve(left_target_pose, right_target_pose, viz=True)
    ik_solver.get_joint_names()

    print(f'{converged=}')


if __name__ == '__main__':
    main()
