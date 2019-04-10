
import os
from os.path import join

import tensorflow as tf

tf.enable_eager_execution()

current_dir = os.path.dirname(__file__)
model_dir = join(current_dir, "log", "example")
output_dir = join(model_dir, "result")
input_dir = join(current_dir, "test")
depth_dir = join(current_dir, "data", "depth")
normal_dir = join(current_dir, "data", "normal")

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
    image_file = join(input_dir, file)
    depth_file = join(depth_dir, file)
    normal_file = join(normal_dir, file)

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


    input_image = load_png(image_file)
    depth_image = load_png(depth_file, gray=True)
    normal_image = load_png(normal_file, hue=True)
    input = tf.expand_dims(tf.concat([input_image, depth_image, normal_image], axis=-1), 0)

    lines, depth = generator(input)

    def prepare(img):
        img = tf.image.resize_images(img, size=[IMG_HEIGHT, IMG_WIDTH],
                           align_corners=True, method=tf.image.ResizeMethod.BICUBIC)
        img = img * 0.5 + 0.5
        img = tf.reshape(img, shape=[IMG_HEIGHT, IMG_WIDTH, 1])
        img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
        img = tf.image.encode_png(img)
        return img

    lines = prepare(lines)
    depth = prepare(depth)
    tf.write_file(os.path.join(output_dir, "lines", file), lines)
    tf.write_file(os.path.join(output_dir, "depth", file), depth)
