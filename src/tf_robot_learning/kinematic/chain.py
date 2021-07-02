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


def stack_batch(vecs):
    """
    Stack a list of vectors

    :param vecs: 	list of dim0 x tf.array((dim1, dim2)) or dim0 x tf.array((dim2,))
    :return: tf.array((dim1, dim0, dim2)) or tf.array(dim0, dim2)
    """
    if vecs[0].shape.ndims < vecs[-1].shape.ndims:  # if first frame is not batch
        vecs[0] = tf.ones((tf.shape(vecs[-1])[0], 1)) * vecs[0]

    if vecs[-1].shape.ndims == 1:
        return tf.stack(vecs)
    elif vecs[-1].shape.ndims == 2:
        return tf.transpose(tf.stack(vecs), perm=(1, 0, 2))
    else:
        return NotImplementedError


def return_frame(p, layout=FkLayout.xm):
    """
    Return a frame or list of frame in the desired layout

    :param p: 		Frame
    :param layout:	FkLayout
    :return:
    """
    if layout is FkLayout.x:
        if isinstance(p, list):
            return stack_batch([_p.p for _p in p])
        else:
            return p.p
    elif layout is FkLayout.xq:
        if isinstance(p, list):
            return stack_batch([_p.xq for _p in p])
        else:
            return p.xq
    elif layout is FkLayout.xm:
        if isinstance(p, list):
            return stack_batch([_p.xm for _p in p])
        else:
            return p.xm
    elif layout is FkLayout.xmv:
        if isinstance(p, list):
            return stack_batch([_p.xmv for _p in p])
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

    def fk(self, q, layout=FkLayout.xm, floating_base=None):
        """
        Pose of all segments of the chain

        :param q:		[batch_size, nb_joint] or [nb_joint] or list of [batch_size] Joint angles
        :param layout: layout of frame
        :param floating_base Frame() or tuple (p translation vector, m rotation matrix)
        :return:
        """
        assert q.shape[1] == self.get_num_joints()

        batch_size = q.shape[0]

        if floating_base is None:
            p = [Frame(batch_shape=batch_size)]
        elif isinstance(floating_base, tuple) or isinstance(floating_base, list):
            p = [Frame(p=floating_base[0], m=floating_base[1], batch_shape=batch_size)]
        elif isinstance(floating_base, Frame):
            p = [floating_base]
        else:
            raise ValueError("Unknown floating base type")

        j = 0

        for i in range(self.nb_segm):
            if self.segments[i].joint.type is not JointType.NoneT:
                if isinstance(q, list) or q.shape.ndims == 1:
                    p += [p[-1] * self.segments[i].pose(q[j], batch_size)]
                elif q.shape.ndims == 2:
                    p += [p[-1] * self.segments[i].pose(q[:, j], batch_size)]
                else:
                    raise NotImplementedError
                j += 1
            else:
                p += [p[-1] * self.segments[i].pose(0., batch_size)]

        return return_frame(p, layout)

    def actuated_joint_names(self):
        if self._names is None or self._names is None:
            self._names = [seg.joint.name for seg in self.segments if seg.joint.type != JointType.NoneT]
        return self._names

    def get_num_joints(self):
        if self._nb_joints is None:
            self._nb_joints = len(self.actuated_joint_names())
        return self._nb_joints
