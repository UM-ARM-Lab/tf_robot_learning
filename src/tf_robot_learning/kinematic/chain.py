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

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as ds

from tf_robot_learning.kinematic.frame import Frame
from tf_robot_learning.kinematic.joint import JointType
from tf_robot_learning.kinematic.utils.layout import FkLayout
from tf_robot_learning.kinematic.utils.plot_utils import axis_equal_3d


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

    :param p: 		tf_kdl.Frame or list of [tf_kdl.Frame]
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
    def __init__(self, segments):
        """
        Defines a kinematic Chain

        :param segments:
        """
        self._segments = segments
        self.nb_segm = len(segments)
        self.nb_joint = len([seg for seg in segments if seg.joint.type != JointType.NoneT])

        self._joint_limits = None
        self._mean_pose = None
        self._masses = None
        self._mass = None
        self._names = None
        self._nb_joints = None

    @property
    def joint_limits(self):
        if self._joint_limits is None:
            self._joint_limits = tf.constant([[seg.joint.limits['low'], seg.joint.limits['up']]
                                              for seg in self.segments if seg.joint.type != JointType.NoneT],
                                             dtype=tf.float32)

        return self._joint_limits

    @property
    def segments(self):
        return self._segments

    def ee_frame(self, q, n=0, layout=FkLayout.xm):
        """
        Pose of last-n segment of the cain

        :param q:		tf.Tensor()
            Joint angles
        :param n:		int
            index from end of segment to get
        :param layout:
            layout of frame
        :return:
        """

        if q.shape.ndims == 1:
            p = self.segments[0].pose(q[0], 1)
        elif q.shape.ndims == 2:
            p = self.segments[0].pose(q[:, 0], q.shape[0])

        j = 1

        for i in range(1, self.nb_segm - n):
            if self.segments[i].joint.type is not JointType.NoneT:
                if q.shape.ndims == 1:
                    p = p * self.segments[i].pose(q[j], 1)
                elif q.shape.ndims == 2:
                    p = p * self.segments[i].pose(q[:, j], q.shape[0])
                else:
                    raise NotImplementedError
                j += 1
            else:
                p = p * self.segments[i].pose(0., q.shape[0])

        return return_frame(p, layout)

    def fk(self, q, layout=FkLayout.xm, floating_base=None):
        """
        Pose of all segments of the chain

        :param q:		[batch_size, nb_joint] or [nb_joint] or list of [batch_size] Joint angles
        :param layout: layout of frame
        :param floating_base Frame() or tuple (p translation vector, m rotation matrix)
        :return:
        """
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

            self._nb_joints = len(self._names)

        return self._names
