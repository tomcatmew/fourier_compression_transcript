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


def compress_fix():
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-102-111/102/102_29.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-102-111/105/105_53.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-60-75/69/69_50.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-60-75/64/64_15.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-113-128/118/118_27.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-81-85/83/83_17.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-01-09/09/09_04.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-113-128/128/128_07.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-141-144/143/143_12.bvh'
    #bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-102-111/104/104_13.bvh'
    bvh_path = '/Volumes/CmuMoCap/cmuconvert-daz-113-128/122/122_03.bvh'
    file_name = os.path.basename(bvh_path).rsplit(".")[0]
    time0 = time.time()
    print("#######################")
    bvh = BVH(bvh_path)
    bvh_util.check_channels_are_supported(bvh.channels)
    scale, np_weights = bvh_util.bvh_weights(bvh)
    np_bvh = get_parameter_history_bvh(bvh)[10:, :]  # skip the first frame
    with open('result/{}_fix.csv'.format(file_name), 'w'):
        pass
    print("path:", bvh_path)
    print("scale:", scale)
    frame_jp0 = get_joint_position_history_bvh(bvh)[10:, :, :]  # joint positions
    np_trg = bvh_util.encode_bvh_to_continuous(np_bvh)
    np_bvh1 = bvh_util.decode_continuous_to_bvh(np_trg)
    print("reconstruction_accuracy: ", numpy.linalg.norm(np_bvh - np_bvh1))
    set_parameter_history_bvh_double(bvh, np_bvh1)
    # bvh.save("result/{}.bvh".format(file_name))
    frame_jp1 = get_joint_position_history_bvh(bvh)
    jnt_diff_ratio = (frame_jp0 - frame_jp1).max() / scale
    print("joint position reconstruction: ", jnt_diff_ratio)
    print("shape of data:", np_trg.shape, np_trg.dtype)
    # bvh_util.discontinuity(np_trg,"result/{}.png".format(file_name))

    cycles = []
    net = None
    for itr in range(11):
        net_new = network.MLP(1 + len(cycles) * 2, np_trg.shape[1], num_hidden_layer=1)
        if net is not None:
            network.copy_net_weights(net_new, net)
        net = net_new
        np_out = network.compress(net, cycles, np_trg, np_weights)
        # cycles.append(network.new_cycle(np_out - np_trg, np_weights))
        cycles.append(itr+1)
        print(cycles)
        # convergene
        np_bvh1 = bvh_util.decode_continuous_to_bvh(np_out.astype(numpy.float64))
        print(np_bvh1.shape)
        set_parameter_history_bvh_double(bvh, np_bvh1)
        # bvh.save("result/{}_{}.bvh".format(file_name, itr))
        frame_jp1 = get_joint_position_history_bvh(bvh)
        jnt_diff_ratio = (frame_jp0 - frame_jp1).max() / scale
        cmp_ratio = np_bvh.size / bvh_util.count_parameters(net)
        bvh_diff_pos = (np_bvh[:, :3] - np_bvh1[:, :3]).max()
        bvh_diff_ang = (np_bvh[:, 3:] - np_bvh1[:, 3:]).max()
        print(itr, cmp_ratio, jnt_diff_ratio, bvh_diff_pos, bvh_diff_ang)
        with open('result/{}_fix.csv'.format(file_name), 'a') as f:
            f.write("{},{},{},{},{}\n".format(itr, cmp_ratio, jnt_diff_ratio,cycles[-1],time.time()-time0))


if __name__ == "__main__":
    compress_fix()
