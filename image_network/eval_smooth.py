
import os
from os.path import join

import tensorflow as tf

import numpy as np

import imageio
imageio.plugins.freeimage.download()

tf.enable_eager_execution()

current_dir = os.path.dirname(__file__)
model_dir = join(current_dir, "example_smooth")
input_dir = join(current_dir, "test")
flow_dir = join(current_dir, "data", "flow")
output_dir = join(model_dir, "result")

IMG_WIDTH = 256
IMG_HEIGHT = 256

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

with open(join(model_dir, "model.json"),"r") as f:
    json_str = f.read()
f.close()
generator = tf.keras.models.model_from_json(json_str)
generator.load_weights(join(model_dir, "model.h5"))

files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
prev=None
prev_depth=None
for file in files:
    print(file)
    image = tf.read_file(os.path.join(input_dir, file))
    image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = (image * 2) - 1

    image = tf.image.resize_images(image, size=[IMG_HEIGHT, IMG_WIDTH],
                                   align_corners=True,
                                   method=tf.image.ResizeMethod.AREA)
    image = tf.expand_dims(image, 0)

    flow_file = os.path.join(flow_dir, os.path.splitext(file)[0] + ".exr")
    flow = imageio.imread(flow_file, format="EXR-FI")
    flow = flow[..., :2]
    flow[..., 0] *= -1
    flow = np.flip(flow, axis=[-1])
    flow = tf.image.resize_images(flow, size=[IMG_HEIGHT, IMG_WIDTH], align_corners=True,
                                  method=tf.image.ResizeMethod.AREA)

    #mask = np.zeros((256, 256, 1), dtype=np.float32)
    #for y in range(256):
    #    for x in range(256):
    #        v, u = flow[y][x]
     #       mask[min(int(y - v), 255)][min(int(x + u), 255)] = 1

    flow = tf.expand_dims(tf.convert_to_tensor(flow, dtype=tf.float32), 0)
    #mask = tf.expand_dims(tf.convert_to_tensor(mask, dtype=tf.float32), 0)

    if prev==None:
        lines, depth = generator(tf.concat([image, tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 6))], axis=-1))
    else:
        warped = tf.contrib.image.dense_image_warp(prev, flow)
        warped_depth = tf.contrib.image.dense_image_warp(prev_depth, flow)

        lines, depth = generator(tf.concat([image, warped, warped_depth], axis=-1))

    prev=lines
    prev_depth=depth

    def prepare(img):
        img = tf.image.resize_images(img, size=[IMG_HEIGHT, IMG_WIDTH],
                           align_corners=True, method=tf.image.ResizeMethod.BICUBIC)
        img = img * 0.5 + 0.5
        img = tf.squeeze(img)
        img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
        img = tf.image.encode_png(img)
        return img

    lines = prepare(lines)
    depth = prepare(depth)
    tf.write_file(os.path.join(output_dir, "lines", file), lines)
    tf.write_file(os.path.join(output_dir, "depth", file), depth)
