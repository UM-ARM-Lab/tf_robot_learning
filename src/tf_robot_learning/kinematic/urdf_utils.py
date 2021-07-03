# tf_robot_learning, a all-around tensorflow library for robotics.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tf_robot_learning.
#
# tf_robot_learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tf_robot_learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_robot_learning. If not, see <http://www.gnu.org/licenses/>.
from typing import Optional, Dict, List

import tensorflow as tf

from tf.transformations import euler_matrix
from tf_robot_learning.kinematic.chain import Chain
from tf_robot_learning.kinematic.frame import Frame
from tf_robot_learning.kinematic.joint import JointType, Joint, Link, SUPPORTED_JOINT_TYPES
from tf_robot_learning.kinematic.segment import Segment
from urdf_parser_py import urdf
from urdf_parser_py.urdf import URDF


def urdf_pose_to_tk_frame(pose: Optional[urdf.Pose]):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]

    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation

    return Frame(p=tf.constant([pos], dtype=tf.float32),
                 m=tf.constant([euler_matrix(*rot)[:3, :3]], dtype=tf.float32),
                 batch_shape=1)


def urdf_joint_to_tk_joint(jnt: urdf.Joint):
    """ tk means 'tensorflow kinematics' """
    origin_frame = urdf_pose_to_tk_frame(jnt.origin)

    if jnt.joint_type == 'revolute':
        axis = tf.constant(jnt.axis, dtype=tf.float32)
        axis = tf.squeeze(tf.matmul(origin_frame.m, tf.expand_dims(axis, -1)), -1)
        return Joint(JointType.RotAxis, origin=origin_frame.p, axis=axis, name=jnt.name, limits=jnt.limit), origin_frame

    if jnt.joint_type == 'fixed' or jnt.joint_type == 'prismatic':
        return Joint(JointType.NoneT, name=jnt.name), origin_frame

    raise NotImplementedError("Unknown joint type: %s." % jnt.joint_type)


def urdf_link_to_tk_link(lnk: urdf.Link):
    if lnk.inertial is not None and lnk.inertial.origin is not None:
        return Link(frame=urdf_pose_to_tk_frame(lnk.inertial.origin), mass=lnk.inertial.mass)
    else:
        return Link(frame=urdf_pose_to_tk_frame(None), mass=1.)


def _add(urdf: URDF, parent: str, segments_map: Dict, chain: List[str]):
    """ DFS iteration """


def urdf_to_chain(urdf, root=None, tip=None):
    root = urdf.get_root() if root is None else root
    segments_map = {}

    chain = None if tip is None else urdf.get_chain(root, tip)[1:]  # A list of strings

    for pair in urdf.child_map:
        for joint_name, child_name in urdf.child_map[parent]:
            # iterate to find the right joint, possibly inefficient but this function is usually called once
            for urdf_joint in urdf.joints:
                if urdf_joint.name == joint_name:
                    if urdf_joint.joint_type not in SUPPORTED_JOINT_TYPES:
                        raise NotImplementedError(f'Unsupported joint {joint_name} of type {urdf_joint.joint_type}')

                    tk_jnt, tk_origin = urdf_joint_to_tk_joint(urdf_joint)

                    tk_lnk = urdf_link_to_tk_link(urdf.link_map[child_name])

                    if parent not in segments_map:
                        segments_map[parent] = []
                    segment = Segment(joint=tk_jnt, f_tip=tk_origin, child_name=child_name, link=tk_lnk)
                    segments_map[parent].append((tk_jnt, segment))

    return Chain(segments_map)


def urdf_from_file(file):
    return URDF.from_xml_file(file)
