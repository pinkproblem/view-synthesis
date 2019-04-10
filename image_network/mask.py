# calculates a [0..1] validity mask for an exr flow image

# a pixel is considered valid if a pixel is moved there according to the flow

import imageio, os
imageio.plugins.freeimage.download()
import numpy as np
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir = join(current_dir, "data", "photo_flow")
output_dir = join(current_dir, "data", "photo_mask")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

files = [file for file in os.listdir(input_dir) if file.endswith(".exr")]

for file in files:
    print(file)

    flow = imageio.imread(os.path.join(input_dir, file))
    flow[..., 1] *= -1
    flow = np.flip(flow[..., :2], axis=-1)

    mask = np.zeros([256, 256], dtype=np.float32)
    for y in range(256):
        for x in range(256):
            v, u = flow[y][x]
            if v == 0 and u == 0:
                continue
            #mask[min(255, int(y + v))][min(255, int(x + u))] = 255
            newY = y + v
            newX = x + u
            topleft = (newX % 1) * (newY % 1)
            topright = (1 - (newX % 1)) * (newY % 1)
            botleft = (newX % 1) * (1 - (newY % 1))
            botright = (1 - (newX % 1)) * (1 - (newY % 1))

            xi, yi = np.clip(int(newX - 0.5), 0, 254), np.clip(int(newY - 0.5), 0, 254)
            mask[yi][xi] += topleft
            mask[yi][xi + 1] += topright
            mask[yi + 1][xi] += botleft
            mask[yi + 1][xi + 1] += botright

    mask = np.clip(mask, 0, 1) * 255

    imageio.imwrite(os.path.join(output_dir, os.path.splitext(file)[0] + ".png"), mask)