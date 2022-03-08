import math
import os
import sys
import matplotlib.pyplot as plt
import numpy

sys.path.append(os.path.join(
    os.path.dirname(__file__), 'external', 'delfem2-python-bindings'))
from delfem2.delfem2 import BVH
from delfem2.delfem2 import \
    mat3_from_euler_angle, \
    cartesian_axis_angle_vector_from_mat3, \
    quaternion_from_mat3, \
    mat3_from_quaternion, \
    euler_angle_from_mat3, \
    mat3_from_cartesian_axis_angle_vector, \
    multiply_quaternion_quaternion

def norm_l2(a, b):
    assert len(a) == len(b)
    sum = 0.
    for i in range(len(a)):
        sum += (a[i] - b[i]) * (a[i] - b[i])
    return math.sqrt(sum)


def bvh_weights(bvh: BVH):
    max_dist = numpy.zeros(len(bvh.bones))
    for bone in bvh.bones:
        ibone_next = bone.parent_bone_idx
        list_parent_bone_idx = []
        while ibone_next != -1:
            list_parent_bone_idx.append(ibone_next)
            ibone_next = bvh.bones[ibone_next].parent_bone_idx
        # print(bone.name, bone.position(), list_parent_bone_idx)
        for ip in list_parent_bone_idx:
            dist = norm_l2(bone.position(), bvh.bones[ip].position())
            max_dist[ip] = max(dist, max_dist[ip])

    weights = numpy.zeros((len(bvh.channels)+1))
    weights[0] = 1
    weights[1] = 1
    weights[2] = 1
    weights[3] = max_dist[0]
    weights[4] = max_dist[0]
    weights[5] = max_dist[0]
    weights[6] = max_dist[0]
    nch = len(bvh.channels) // 3
    for ich in range(2,nch):
        ibone = bvh.channels[ich*3].ibone
        weights[ich*3+0+1] = max_dist[ibone]
        weights[ich*3+1+1] = max_dist[ibone]
        weights[ich*3+2+1] = max_dist[ibone]

    bb = bvh.minmax_xyz()
    # print(bb)
    scale = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
    return scale, weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def discontinuity(np0: numpy.ndarray, png_path):
    n = np0.shape[0]
    diff = numpy.ndarray([n - 1, 3])
    for i in range(n - 1):
        d = np0[i + 1] - np0[i]
        d0 = d[:3]
        d1 = d[3:6]
        d2 = d[6:]
        diff[i, 0] = numpy.linalg.norm(d0)
        diff[i, 1] = numpy.linalg.norm(d1)
        diff[i, 2] = numpy.linalg.norm(d2)
    vmax = diff.max()
    # print(diff,vmax)
    fig, ax = plt.subplots()
    ax.plot(diff[:, 0], label="root_pos")
    ax.plot(diff[:, 1], label="root_rot")
    ax.plot(diff[:, 2], label="child_rot")
    # ax.set_xlabel("compression ratio")
    # ax.set_ylabel("Euler angle error in degree")
    ax.legend(loc=0)
    # plt.show()
    plt.savefig(png_path)
    plt.pause(1)



def check_channels_are_supported(chs):
    nch = len(chs)
    assert nch % 3 == 0
    for ich in range(0, nch // 3):
        if ich == 0:
            assert chs[0].is_rot is False
            assert chs[1].is_rot is False
            assert chs[2].is_rot is False
            assert chs[0].iaxis == 0
            assert chs[1].iaxis == 1
            assert chs[2].iaxis == 2
        elif ich == 1:
            assert chs[3].is_rot is True
            assert chs[4].is_rot is True
            assert chs[5].is_rot is True
            assert chs[3].iaxis == 2
            assert chs[4].iaxis == 1
            assert chs[5].iaxis == 0
        else:
            assert chs[ich * 3 + 0].is_rot is True
            assert chs[ich * 3 + 1].is_rot is True
            assert chs[ich * 3 + 2].is_rot is True
            assert chs[ich * 3 + 0].iaxis == 2
            assert chs[ich * 3 + 1].iaxis == 0
            assert chs[ich * 3 + 2].iaxis == 1


def encode_bvh_to_continuous(v: numpy.ndarray):
    assert v.shape[1] % 3 == 0
    nch = v.shape[1] // 3
    vo = numpy.zeros((v.shape[0], v.shape[1] + 1))
    for iframe in range(v.shape[0]):
        # root translation
        vo[iframe, 0] = v[iframe, 0]
        vo[iframe, 1] = v[iframe, 1]
        vo[iframe, 2] = v[iframe, 2]
        # root rotation
        rot1 = mat3_from_euler_angle(
            [v[iframe, 3] * math.pi / 180.0,
             v[iframe, 4] * math.pi / 180.0,
             v[iframe, 5] * math.pi / 180.0],
            [2, 1, 0])
        if iframe > 0:
            rot0 = mat3_from_euler_angle(
                [v[iframe - 1, 3] * math.pi / 180.0,
                 v[iframe - 1, 4] * math.pi / 180.0,
                 v[iframe - 1, 5] * math.pi / 180.0],
                [2, 1, 0])
            rot01 = rot1 @ rot0.transpose()
            quat01 = quaternion_from_mat3(rot01)
            quat0 = vo[iframe - 1, 3:7]
            quat1 = multiply_quaternion_quaternion(quat01, quat0)
        else:
            quat1 = quaternion_from_mat3(rot1)
        vo[iframe, 3] = quat1[0]
        vo[iframe, 4] = quat1[1]
        vo[iframe, 5] = quat1[2]
        vo[iframe, 6] = quat1[3]
        for ich in range(2, nch):
            rot1 = mat3_from_euler_angle(
                [v[iframe, ich * 3 + 0] * math.pi / 180.0,
                 v[iframe, ich * 3 + 1] * math.pi / 180.0,
                 v[iframe, ich * 3 + 2] * math.pi / 180.0],
                [2, 0, 1])
            aav0 = cartesian_axis_angle_vector_from_mat3(rot1)
            vo[iframe, ich * 3 + 0 + 1] = aav0[0]
            vo[iframe, ich * 3 + 1 + 1] = aav0[1]
            vo[iframe, ich * 3 + 2 + 1] = aav0[2]
    return vo


def decode_continuous_to_bvh(vo: numpy.ndarray):
    assert (vo.shape[1] - 1) % 3 == 0
    nch = (vo.shape[1] - 1) // 3
    v = numpy.zeros((vo.shape[0], vo.shape[1] - 1))
    for iframe in range(vo.shape[0]):
        # root translation
        v[iframe, 0] = vo[iframe, 0]
        v[iframe, 1] = vo[iframe, 1]
        v[iframe, 2] = vo[iframe, 2]
        # root rotation
        q = vo[iframe, 3:7].copy()
        if numpy.linalg.norm(q) < 1.0e-10:
            q = numpy.array([0,0,0,1])
        else:
            q /= numpy.linalg.norm(q)
        rot0 = mat3_from_quaternion(q)
        ea0 = euler_angle_from_mat3(rot0, [2, 1, 0])
        v[iframe, 3] = ea0[0] * 180 / math.pi
        v[iframe, 4] = ea0[1] * 180 / math.pi
        v[iframe, 5] = ea0[2] * 180 / math.pi
        for ich in range(2, nch):
            aav = vo[iframe, ich * 3 + 1:ich * 3 + 4].copy()
            rot0 = mat3_from_cartesian_axis_angle_vector(aav)
            ea0 = euler_angle_from_mat3(rot0, [2, 0, 1])
            v[iframe, ich * 3 + 0] = ea0[0] * 180 / math.pi
            v[iframe, ich * 3 + 1] = ea0[1] * 180 / math.pi
            v[iframe, ich * 3 + 2] = ea0[2] * 180 / math.pi
    return v


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "asset", "walk.bvh")
    bvh = BVH()
    bvh.open(path)
    # bvh.clear_pose()
    scale, weights = bvh_weights(bvh)
    print("skeleton_size", scale, weights)
