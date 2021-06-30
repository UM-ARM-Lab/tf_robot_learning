import pathlib
from math import pi
from time import sleep

import PyKDL as kdl
import tensorflow as tf

import kdl_parser_py.urdf as kdl_parser
import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from geometry_msgs.msg import Point
from moveit_msgs.msg import DisplayRobotState
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, tk_tree_from_urdf_model, kdl_chain_from_urdf_model
from tf_robot_learning.kinematic.utils.layout import FkLayout
from visualization_msgs.msg import Marker


def main():
    rospy.init_node("fk_demo")

    pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
    pub2 = get_connected_publisher("point", Marker, queue_size=10)

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    urdf = urdf_from_file(urdf_filename.as_posix())

    left = kdl_chain_from_urdf_model(urdf, tip='left_tool')
    right = kdl_chain_from_urdf_model(urdf, tip='right_tool')
    # chain = tk_tree_from_urdf_model(urdf)

    robot = DisplayRobotState()
    joint_names = list(set(left.actuated_joint_names() + right.actuated_joint_names()))
    robot.state.joint_state.name = joint_names

    while True:
        left_q = tf.random.uniform([len(left.actuated_joint_names())], -pi, pi, dtype=tf.float32) * 0.01
        right_q = tf.random.uniform([len(right.actuated_joint_names())], -pi, pi, dtype=tf.float32) * 0.01
        # q = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        left_xs = left.xs(left_q, layout=FkLayout.xm)
        right_xs = right.xs(right_q, layout=FkLayout.xm)
        positions = tf.concat([left_xs[:, :3], right_xs[:, :3]], axis=0)
        # positions = left_xs[:, :3]

        # robot.state.joint_state.position = q.numpy().tolist()
        # pub.publish(robot)

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
        sleep(1)


def joints_to_kdl(positions):
    pos_array = kdl.JntArray(len(positions))
    for i, p in enumerate(positions):
        pos_array[i] = p
    return pos_array


def main2():
    # load in ros parameters
    baselink = "hdt_michigan_root"
    endlink = "left_tool"
    flag, tree = kdl_parser.treeFromParam("/robot_description")

    # build kinematic chain and fk and jacobian solvers
    chain_ee = tree.getChain(baselink, endlink)
    fk_ee = kdl.TreeFkSolverPos(chain_ee)
    dir(kdl)

    end_frame = kdl.Frame()
    joint_values = [0] * 9
    _fk_p_kdl = kdl.ChainFkSolverPos_recursive(chain_ee)
    _fk_p_kdl.JntToCart(joints_to_kdl(joint_values), end_frame)
    end_frame


if __name__ == '__main__':
    main()
