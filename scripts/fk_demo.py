import pathlib

import tensorflow as tf

import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from moveit_msgs.msg import DisplayRobotState
from tf_robot_learning import kinematic as tk


def main():
    rospy.init_node("fk_demo")

    pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    urdf = tk.urdf_from_file(urdf_filename.as_posix())

    chain = tk.kdl_chain_from_urdf_model(urdf, tip='left_tool')

    q = tf.zeros(40, dtype=tf.float32)
    xs = chain.xs(q, layout=tk.FkLayout.xm)

    msg = DisplayRobotState()
    msg.state.joint_state.name = joint_names
    msg.state.joint_state.position = q
    pub.publish(msg)


if __name__ == '__main__':
    main()
