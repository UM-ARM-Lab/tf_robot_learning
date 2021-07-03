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
from link_bot_classifiers.robot_points import RobotVoxelgridInfo
from moonshine.simple_profiler import SimpleProfiler
from moveit_msgs.msg import DisplayRobotState
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler
from tf_robot_learning.kinematic.joint import SUPPORTED_ACTUATED_JOINT_TYPES
from tf_robot_learning.kinematic.tree import Tree
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, urdf_to_tree
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


def compute_pose_loss(ee_pose, target_pose):
    position = ee_pose[:, :3]
    orientation = tf.reshape(ee_pose[:, 3:], [-1, 3, 3])
    target_position = target_pose[:, :3]
    target_quat = target_pose[:, 3:]
    target_orientation = tfr.from_quaternion(target_quat)
    _orientation_error = orientation_error_mat(target_orientation, orientation)
    position_error = tf.reduce_sum(tf.square(position - target_position), axis=-1)

    return position_error, _orientation_error


def compute_jl_loss(tree: Tree, q):
    joint_limits = tree.get_joint_limits()
    jl_low = joint_limits[:, 0][tf.newaxis]
    jl_high = joint_limits[:, 1][tf.newaxis]
    low_error = tf.math.maximum(jl_low - q, 0)
    high_error = tf.math.maximum(q - jl_high, 0)
    jl_errors = tf.math.maximum(low_error, high_error)
    jl_loss = tf.reduce_sum(jl_errors, axis=-1)
    return jl_loss


def target(x, y, z, roll, pitch, yaw):
    return tf.cast(tf.expand_dims(tf.concat([[x, y, z], quaternion_from_euler(roll, pitch, yaw)], 0), 0), tf.float32)


class HdtIK:

    def __init__(self, urdf_filename: pathlib.Path, max_iters: int = 5000):
        self.urdf = urdf_from_file(urdf_filename.as_posix())

        self.tree = urdf_to_tree(self.urdf)
        self.left_ee_name = 'left_tool'
        self.right_ee_name = 'right_tool'

        self.actuated_joint_names = list([j.name for j in self.urdf.joints if j.type in SUPPORTED_ACTUATED_JOINT_TYPES])
        self.n_actuated_joints = len(self.actuated_joint_names)

        self.robot_info = RobotVoxelgridInfo(joint_positions_key='!!!')

        self.max_iters = max_iters
        self.theta = 0.995
        self.jl_alpha = 0.1
        self.initial_lr = 0.01
        self.loss_threshold = 1e-4

        # lr = tf.keras.optimizers.schedules.ExponentialDecay(0.5, 20, 0.9)
        # opt = tf.keras.optimizers.SGD(lr)
        self.optimizer = tf.keras.optimizers.Adam(self.initial_lr, amsgrad=True)

        self.display_robot_state_pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
        self.point_pub = get_connected_publisher("point", Marker, queue_size=10)
        self.joint_states_viz_pub = rospy.Publisher("joint_states_viz", JointState, queue_size=10)
        self.tf2 = TF2Wrapper()

        self.p = SimpleProfiler()

    def solve(self, env_points, left_target_pose, right_target_pose, initial_value=None, viz=False):
        if initial_value is None:
            batch_size = left_target_pose.shape[0]
            initial_value = tf.zeros([batch_size, self.get_num_joints()], dtype=tf.float32)
        q = tf.Variable(initial_value)

        converged = False
        for _ in trange(self.max_iters):
            loss, gradients, viz_info = self.opt(q, env_points, left_target_pose, right_target_pose)
            if loss < self.loss_threshold:
                converged = True
                break

            if viz:
                self.viz_func(left_target_pose, right_target_pose, q, viz_info)

        return q, converged

    def print_stats(self):
        print(self.p)

    def opt(self, q, env_points, left_target_pose, right_target_pose):
        with tf.GradientTape() as tape:
            self.p.start()
            loss, viz_info = self.step(q, env_points, left_target_pose, right_target_pose)
            self.p.stop()
        gradients = tape.gradient([loss], [q])
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, [q]))
        return loss, gradients, viz_info

    # @tf.function
    def step(self, q, env_points, left_target_pose, right_target_pose):
        poses = self.tree.fk(q)
        jl_loss = self.compute_jl_loss(self.tree, q)
        left_ee_pose = poses[self.left_ee_name]
        right_ee_pose = poses[self.right_ee_name]
        left_pose_loss = self.compute_pose_loss(left_ee_pose, left_target_pose)
        right_pose_loss = self.compute_pose_loss(right_ee_pose, right_target_pose)

        # # FIXME: fix the FK code to handle trees better, to avoid duplicating the computation
        # #  and so that we can get the gripper links when we call FK.
        # #  run depth-first iteration accumulating matrix products?
        def _compute_collision_loss():
            # # collision_loss = self.compute_collision_loss(left_xs, right_xs, env_points)
            # # compute robot points given q
            # link_transforms = {}
            # for left_link_i, left_link in enumerate(self.left.segments):
            # left_link_x = left_xs[:, left_link_i + 1]
            # link_transforms[left_link.child_name] = left_link_x
            # for right_link_i, right_link in enumerate(self.right.segments):
            # right_link_x = right_xs[:, right_link_i + 1]
            # link_transforms[right_link.child_name] = right_link_x
            # link_to_robot_transforms = []
            # for link_name in self.robot_info.link_names:
            # link_to_robot_transform = link_transforms[link_name]
            # link_to_robot_transforms.append(link_to_robot_transform)
            # # [b, n_links, 4, 4, 1], links/order based on robot_info
            # link_to_robot_transforms = tf.stack(link_to_robot_transforms, axis=0)
            # links_to_robot_transform_batch = tf.repeat(link_to_robot_transforms, self.robot_info.points_per_links, axis=1)
            # batch_size = q.shape[0]
            # points_link_frame_homo_batch = repeat_tensor(self.robot_info.points_link_frame, batch_size, 0, True)
            # points_robot_frame_homo_batch = tf.matmul(links_to_robot_transform_batch, points_link_frame_homo_batch)
            # points_robot_frame_batch = points_robot_frame_homo_batch[:, :, :3, 0]
            # # compute the distance matrix between robot points and the environment points
            pass

        losses = [
            left_pose_loss,
            right_pose_loss,
            jl_loss,
            # collision_loss,
        ]
        loss = tf.reduce_mean(tf.math.add_n(losses))

        viz_info = [poses]

        return loss, viz_info

    def compute_jl_loss(self, tree: Tree, q):
        return self.jl_alpha * compute_jl_loss(tree, q)

    def compute_pose_loss(self, xs, target_pose):
        pos_error, rot_error = compute_pose_loss(xs, target_pose)
        pose_loss = self.theta * pos_error + (1 - self.theta) * rot_error
        return pose_loss

    def viz_func(self, left_target_pose, right_target_pose, q, viz_info):
        poses, = viz_info
        b = 0
        self.tf2.send_transform(left_target_pose[b, :3].numpy().tolist(),
                                left_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='left_target')
        self.tf2.send_transform(right_target_pose[b, :3].numpy().tolist(),
                                right_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='right_target')

        robot_state_dict = {}
        for name, pose in zip(self.actuated_joint_names, q[b].numpy().tolist()):
            robot_state_dict[name] = pose

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
        msg.points = []
        for pose in poses.values():
            position = pose.numpy()[b, :3]
            p = Point(x=position[0], y=position[1], z=position[2])
            msg.points.append(p)

        self.point_pub.publish(msg)

    def get_joint_names(self):
        return self.actuated_joint_names

    def get_num_joints(self):
        return self.n_actuated_joints


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    ik_solver = HdtIK(urdf_filename, max_iters=500)

    batch_size = 32
    viz = True

    left_target_pose = tf.tile(target(-0.3, 0.6, 0.2, 0, -pi / 2, -pi / 2), [batch_size, 1])
    right_target_pose = tf.tile(target(0.3, 0.6, 0.2, -pi / 2, -pi / 2, 0), [batch_size, 1])
    env_points = tf.random.uniform([batch_size, 10, 3], -1, 1, dtype=tf.float32)

    initial_value = tf.zeros([batch_size, ik_solver.get_num_joints()], dtype=tf.float32)
    q, converged = ik_solver.solve(env_points=env_points,
                                   left_target_pose=left_target_pose,
                                   right_target_pose=right_target_pose,
                                   viz=viz,
                                   initial_value=initial_value)
    ik_solver.get_joint_names()
    print(f'{converged=}')
    ik_solver.print_stats()


if __name__ == '__main__':
    main()
