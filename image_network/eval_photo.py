
import os
from os.path import join

import tensorflow as tf

import imageio
imageio.plugins.freeimage.download()

tf.enable_eager_execution()


current_dir = os.path.dirname(__file__)
model_dir = join(current_dir, "log", "example_photo")
output_dir = join(model_dir, "result")
input_dir = join(current_dir, "test")
depth_dir = join(current_dir, "data", "depth")
normal_dir = join(current_dir, "data", "normal")
photo_dir = join(current_dir, "data", "photo")
flow_dir =join(current_dir, "data", "photo_flow")
mask_dir =join(current_dir, "data", "photo_mask")

IMG_WIDTH = 256
IMG_HEIGHT = 256

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

os.mkdir(join(output_dir, "lines"))
os.mkdir(join(output_dir, "depth"))

with open(join(model_dir, "model.json"),"r") as f:
    json_str = f.read()
f.close()
generator = tf.keras.models.model_from_json(json_str)
generator.load_weights(join(model_dir, "model.h5"))

files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
for file in files:
    print(file)

    basename = os.path.splitext(os.path.basename(file))[0]
    depth_file = join(depth_dir, basename + ".png")
    normal_file = join(normal_dir, basename + ".png")
    photo_file = join(photo_dir, basename + ".png")
    flow_file = join(flow_dir, basename + ".exr")
    mask_file = join(mask_dir, basename + ".png")

    def load_png(file, gray=False, hue=False):
        image = tf.read_file(file)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=[IMG_HEIGHT, IMG_WIDTH], align_corners=True)
        if gray:
            image = tf.image.rgb_to_grayscale(image)
        elif hue:
            image = tf.image.rgb_to_hsv(image)
            image = tf.slice(image, [0, 0, 0], [IMG_HEIGHT, IMG_WIDTH, 1])
        image = (image * 2) - 1
        return image

    def load_exr(path):
        flow = imageio.imread(path, format="EXR-FI")
        flow = flow[..., :2]
        flow[..., 0] *= -1
        flow = tf.convert_to_tensor(flow)
        return tf.reverse(flow, axis=[-1])


    input_image = load_png(os.path.join(input_dir, file))
    depth_image = load_png(depth_file, gray=True)
    normal_image = load_png(normal_file, hue=True)
    photo_image = load_png(photo_file, gray=True)

    flow = load_exr(flow_file)

    mask = tf.read_file(mask_file)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
    mask = tf.image.resize_images(mask, size=[IMG_HEIGHT, IMG_WIDTH], align_corners=True)
    mask = tf.image.rgb_to_grayscale(mask)

    warped_photo = tf.reshape(tf.contrib.image.dense_image_warp(tf.expand_dims(photo_image, 0), flow), [IMG_HEIGHT, IMG_WIDTH, 1])

    lines, depth = generator(tf.expand_dims(tf.concat([input_image, depth_image, normal_image, warped_photo, mask], axis=-1),0))

    def prepare(img):
        img = tf.image.resize_images(img, size=[IMG_HEIGHT, IMG_WIDTH],
                           align_corners=True, method=tf.image.ResizeMethod.BICUBIC)
        img = img * 0.5 + 0.5
        img = tf.expand_dims(tf.squeeze(img), axis=-1)
        img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
        img = tf.image.encode_png(img)
        return img

    lines = prepare(lines)
    depth = prepare(depth)
    tf.write_file(os.path.join(output_dir, "lines", file), lines)
    tf.write_file(os.path.join(output_dir, "depth", file), depth)
