from plyfile import PlyData
import os
import numpy as np
import random

current_dir = os.path.dirname(__file__)
input_dir = current_dir
output_dir = current_dir

voxel_num = 32 # cube side length
points_per_voxel = 32 # any further points are discarded
point_dim = 9 # x, y, z, dx, dy, dz, r, g, b

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for file in os.listdir(input_dir):
    if not file.endswith(".ply"):
        continue

    print(file)

    file_path = os.path.join(input_dir, file)
    name = file[:-4]

    vertices = [[[[] for k in range(voxel_num)] for j in range(voxel_num)] for i in range(voxel_num)]
    cube = np.zeros([voxel_num, voxel_num, voxel_num, points_per_voxel, point_dim])
    plydata = PlyData.read(file_path)

    shift = np.asarray((0.5, 0.1, 0.5))

    skipped = 0
    for vertex in plydata['vertex']:
        position = np.asarray((vertex['x'], vertex['y'], vertex['z']))
        position += shift
        color = np.asarray((vertex['red']/255, vertex['green']/255, vertex['blue']/255))
        index = (position * voxel_num).astype(int)
        if (index < 0).any() or (index > voxel_num - 1).any():
            skipped += 1
            continue

        stacked = np.concatenate([position, color], axis=-1)
        vertices[index[0]][index[1]][index[2]].append(stacked)

    for idx in np.ndindex(voxel_num, voxel_num, voxel_num):
        sublist = np.array(vertices[idx[0]][idx[1]][idx[2]])
        if len(sublist) == 0:
            sublist = np.zeros((1, point_dim - 3))
        random.shuffle(sublist)
        sublist = sublist[:min(points_per_voxel, len(sublist))]

        pos_list = sublist[..., 0:3]
        centroid = np.mean(pos_list, axis=0)
        rel_pos = pos_list - np.tile(centroid, (len(pos_list), 1))
        sublist = np.concatenate([sublist, rel_pos], axis=-1)

        fill_zeros = np.tile(np.zeros((point_dim, 1)), max(0, points_per_voxel - len(sublist))).transpose()
        sublist = np.concatenate([sublist, fill_zeros], axis=0)

        cube[idx[2]][idx[1]][idx[0]] = sublist


    print("skipped " + str(skipped) + " / " + str(len(plydata['vertex']['x'])))
    np.save(os.path.join(output_dir, name), cube)

