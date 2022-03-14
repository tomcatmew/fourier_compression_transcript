import os.path
import sys
import numpy

sys.path.append(os.path.join(
    os.path.dirname(__file__), 'external', 'delfem2-python-bindings'))
print(__file__)
print(sys.path)
import delfem2
from delfem2.delfem2 import \
    BVH, \
    get_meshtri3_rigbones_octahedron, \
    get_joint_position_history_bvh
import delfem2.mesh

def ExportFramesTrajectory(path:str):
    bvh = BVH(path)
    for ib in range(len(bvh.bones)):
        print(ib, bvh.bones[ib].name)
    file_name = os.path.basename(path).rsplit('.')[0]

    for iframe in range(0,bvh.nframe,100):
        bvh.set_frame(iframe)
        V, F = get_meshtri3_rigbones_octahedron(bvh)
        delfem2.mesh.write_uniform_mesh("result/{}_frame_{}.obj".format(file_name,iframe), V, F)

    joint_positions = get_joint_position_history_bvh(bvh)

    list_VE = []
#    for ind_j in {0, 5, 11, 18, 23, 32}:
    for ind_j in {51,56}:
        V = joint_positions[:, ind_j, :].copy()
        E = numpy.array([[i, i + 1] for i in range(joint_positions.shape[0] - 1)])
        list_VE.append( [V,E] )
    V, E = delfem2.mesh.concat(list_VE)
    delfem2.mesh.write_uniform_mesh("result/{}_traj.obj".format(file_name), V, E)


if __name__ == "__main__":
    ExportFramesTrajectory("result/138_02.bvh")
    # ExportFramesTrajectory("result/127_26_5.bvh")
