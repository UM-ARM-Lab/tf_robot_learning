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
from typing import List

import tensorflow as tf

from tf_robot_learning.kinematic.frame import Frame
from tf_robot_learning.kinematic.joint import JointType
from tf_robot_learning.kinematic.segment import Segment
from tf_robot_learning.kinematic.utils.layout import FkLayout


def return_frame(p, layout=FkLayout.xm):
    """
    Return a frame or list of frame in the desired layout

    :param p: 		Frame
    :param layout:	FkLayout
    :return:
    """
    if layout is FkLayout.x:
        if isinstance(p, list):
            return tf.stack([_p.p for _p in p], axis=1)
        else:
            return p.p
    elif layout is FkLayout.xq:
        if isinstance(p, list):
            return tf.stack([_p.xq for _p in p], axis=1)
        else:
            return p.xq
    elif layout is FkLayout.xm:
        if isinstance(p, list):
            return tf.stack([_p.xm for _p in p], axis=1)
        else:
            return p.xm
    elif layout is FkLayout.xmv:
        if isinstance(p, list):
            return tf.stack([_p.xmv for _p in p], axis=1)
        else:
            return p.xmv
    elif layout is FkLayout.f:
        return p


class Chain:
    def __init__(self, segments: List[Segment]):
        self._segments = segments
        self.nb_segm = len(segments)

        self._joint_limits = None
        self._names = None
        self._nb_joints = None

    def get_joint_limits(self):
        if self._joint_limits is None:
            self._joint_limits = tf.constant([[seg.joint.limits['low'], seg.joint.limits['up']]
                                              for seg in self.segments if seg.joint.type != JointType.NoneT],
                                             dtype=tf.float32)

        return self._joint_limits

    @property
    def segments(self):
        return self._segments

    def fk(self, q, layout=FkLayout.xm):
        """
        Pose of all segments of the chain

        :param q:		[batch_size, nb_joint] or [nb_joint] or list of [batch_size] Joint angles
        :param layout: layout of frame
        :return:
        """
        assert q.shape[1] == self.get_num_joints()

        batch_size = q.shape[0]

        p = [Frame(batch_shape=batch_size)]  # could use this to represent transform from robot to world

        for segment in self.segments:
            if segment.joint.type is not JointType.NoneT:
                j = self.actuated_joint_names().index(segment.joint.name)
                if isinstance(q, list) or q.shape.ndims == 1:
                    p.append(p[-1] * segment.pose(q[j], batch_size))
                elif q.shape.ndims == 2:
                    p.append(p[-1] * segment.pose(q[:, j], batch_size))
                else:
                    raise NotImplementedError()
            else:
                p.append(p[-1] * segment.pose(tf.zeros([1]), 1))

        return return_frame(p, layout)

    def actuated_joint_names(self):
        if self._names is None or self._names is None:
            self._names = [seg.joint.name for seg in self.segments if seg.joint.type != JointType.NoneT]
        return self._names

    def get_num_joints(self):
        if self._nb_joints is None:
            self._nb_joints = len(self.actuated_joint_names())
        return self._nb_joints
