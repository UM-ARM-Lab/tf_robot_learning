import pathlib
from math import pi
from time import sleep

import tensorflow as tf

import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from geometry_msgs.msg import Point
from moveit_msgs.msg import DisplayRobotState
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, kdl_chain_from_urdf_model
from tf_robot_learning.kinematic.utils.layout import FkLayout
from visualization_msgs.msg import Marker


def main():
    rospy.init_node("fk_demo")

    pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
    pub2 = get_connected_publisher("point", Marker, queue_size=10)

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    urdf = urdf_from_file(urdf_filename.as_posix())

    chain = kdl_chain_from_urdf_model(urdf, tip='left_tool')

    robot = DisplayRobotState()
    robot.state.joint_state.name = chain.actuated_joint_names()

    while True:
        # q = tf.random.uniform([len(joint_names)], -pi, pi, dtype=tf.float32)
        q = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        xs = chain.xs(q, layout=FkLayout.xm)
        positions = xs[:, :3]
        rotations = tf.reshape(xs[:, 3:], [-1, 3, 3])

        robot.state.joint_state.position = q.numpy().tolist()
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
        sleep(1)


if __name__ == '__main__':
    main()
