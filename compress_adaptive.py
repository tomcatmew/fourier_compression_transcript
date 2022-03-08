import glob
import os
import random
import sys
import numpy
import time

sys.path.append(os.path.join(
    os.path.dirname(__file__), 'external', 'delfem2-python-bindings'))
from delfem2.delfem2 import BVH
from delfem2.delfem2 import get_parameter_history_bvh
from delfem2.delfem2 import get_joint_position_history_bvh
from delfem2.delfem2 import set_parameter_history_bvh_double

import bvh_util
import network


def compress_cmu(path_dir):
    bvh_paths = glob.glob(path_dir + '/*/*/*.bvh')
    random.shuffle(bvh_paths)
    # bvh_paths = sorted(bvh_paths)
    print("path_dir: ", path_dir)
    print("num_bvh: ", len(bvh_paths))
    print("bvh_paths: ", bvh_paths)
    for bvh_path in bvh_paths:
        # bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-102-111/102/102_29.bvh'
        # bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-102-111/105/105_53.bvh'
        # bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-113-128/127/127_31.bvh'
        file_name = os.path.basename(bvh_path).rsplit(".")[0]
        time0 = time.time()
        print("#######################")
        bvh = BVH(bvh_path)
        bvh_util.check_channels_are_supported(bvh.channels)
        scale, np_weights = bvh_util.bvh_weights(bvh)
        np_bvh = get_parameter_history_bvh(bvh)[10:, :]  # skip the first frame
        if np_bvh.shape[0] > 900:
            continue
        with open('result/{}.csv'.format(file_name), 'w'):
            pass
        print("path:", bvh_path)
        print("scale:", scale)
        frame_jp0 = get_joint_position_history_bvh(bvh)[10:, :, :]  # joint positions
        np_trg = bvh_util.encode_bvh_to_continuous(np_bvh)
        np_bvh1 = bvh_util.decode_continuous_to_bvh(np_trg)
        print("reconstruction_accuracy: ", numpy.linalg.norm(np_bvh - np_bvh1))
        set_parameter_history_bvh_double(bvh, np_bvh1)
        bvh.save("result/{}.bvh".format(file_name))
        frame_jp1 = get_joint_position_history_bvh(bvh)
        jnt_diff_ratio = (frame_jp0 - frame_jp1).max() / scale
        print("joint position reconstruction: ", jnt_diff_ratio)
        print("shape of data:", np_trg.shape, np_trg.dtype)
        bvh_util.discontinuity(np_trg,"result/{}.png".format(file_name))

        cycles = []
        net = None
        for itr in range(11):
            net_new = network.MLP(1 + len(cycles) * 2, np_trg.shape[1], num_hidden_layer=1)
            if net is not None:
                network.copy_net_weights(net_new, net)
            net = net_new
            np_out = network.compress(net, cycles, np_trg, np_weights)
            cycles.append(network.new_cycle(np_out - np_trg, np_weights))
            # convergene
            np_bvh1 = bvh_util.decode_continuous_to_bvh(np_out.astype(numpy.float64))
            print(np_bvh1.shape)
            set_parameter_history_bvh_double(bvh, np_bvh1)
            bvh.save("result/{}_{}.bvh".format(file_name, itr))
            frame_jp1 = get_joint_position_history_bvh(bvh)
            jnt_diff_ratio = (frame_jp0 - frame_jp1).max() / scale
            cmp_ratio = np_bvh.size / bvh_util.count_parameters(net)
            bvh_diff_pos = (np_bvh[:, :3] - np_bvh1[:, :3]).max()
            bvh_diff_ang = (np_bvh[:, 3:] - np_bvh1[:, 3:]).max()
            print(itr, cmp_ratio, jnt_diff_ratio, bvh_diff_pos, bvh_diff_ang)
            with open('result/{}.csv'.format(file_name), 'a') as f:
                f.write("{},{},{},{},{}\n".format(itr, cmp_ratio, jnt_diff_ratio,cycles[-1],time.time()-time0))
            print(cycles)

if __name__ == "__main__":
    path_dir = '/Volumes/CmuMoCap'
    compress_cmu(path_dir)
