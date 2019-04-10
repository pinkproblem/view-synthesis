
import os, imageio, pyrr
import numpy as np
from plyfile import PlyData, PlyElement
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir = join(current_dir, "log", "example", "result", "lines")
depth_dir = join(current_dir, "log", "example", "result", "depth")
output_dir = join(current_dir, "log", "example", "result")

vertices = []


files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
for i, file in enumerate(files):
    print(file)
        
    # get corresponding
    name = os.path.splitext(file)[0]
    depth_file = os.path.join(depth_dir, name + ".png")
    if not os.path.isfile(depth_file):
        print("No corresponding depth file found for " + name)
        continue
        
    im = imageio.imread(os.path.join(input_dir, file))
    z_im = imageio.imread(depth_file)

    # scale depth img to [0, 1]
    z_im = (z_im / 255) + 1.5
    #z_im = np.clip(z_im, 0.1, 100)
    
    radius = 2
    center = np.array([0, 0.5, 0])
    angle = 2 * np.pi - i * 2 * np.pi / 100
    pos = np.array([radius * np.sin(angle), 0.5, radius * np.cos(angle)])

    #fov = 2 * np.arctan((0.5 * 300) / (0.5 * 400 / np.tan(49.1 * np.pi/180 / 2)))
    
    mv = pyrr.matrix44.create_look_at(eye=pos, target=center, up=np.array([0, 1, 0])).transpose()
    #p = pyrr.matrix44.create_perspective_projection(fovy=fov*180/np.pi, aspect=400/300, near=0.1, far=100).transpose()

    p = np.array([(2.1875, 0.0000,  0.0000,  0.0000),
            (0.0000, 2.9167,  0.0000,  0.0000),
            (0.0000, 0.0000, -1.0020, -0.2002),
            (0.0000, 0.0000, -1.0000,  0.0000)])

    mv_i = np.linalg.inv(mv)
    p_i = np.linalg.inv(p)
    
    for y, row in enumerate(im):
        for x, pixel in enumerate(row):
            z_raw = z_im[y][x][0]
            if np.max(pixel[:3]) < 100:
                A = p[2][2]
                B = p[2][3]
                z = (-A * z_raw + B) / z_raw
                pi = np.array([(x-200)/200, -(y-150)/150, z, 1])
                pc = np.dot(p_i, pi)
                pc = pc / pc[3]
                pw = np.dot(mv_i, pc)

                vertices.append(tuple(pw[0:3]))
                
vertex = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
ply.write("out.ply")
    