import tensorflow as tf
import numpy as np
import pyrr
from os.path import join
import os
import custom_layers
import imageio

tf.enable_eager_execution()

current_dir = os.path.dirname(__file__)
model_file = join(current_dir, "01.npy")
log_dir = join(current_dir, "log", "example")
name = "eval"

build_gif = True

model = np.load(model_file)
model = tf.convert_to_tensor(model, dtype=tf.float32)
model = tf.reshape(model, [1, 32, 32, 32, 32, 9])


custom_objects = {'ElementwiseMaxPool': custom_layers.ElementwiseMaxPool
    , 'ReduceSum': custom_layers.ReduceSum
    , 'RepeatElements': custom_layers.RepeatElements
    , 'ConcatDepth': custom_layers.ConcatDepth
    , 'ProjectionLayer': custom_layers.ProjectionLayer}
with open(join(log_dir, "model.json"),"r") as f:
    json_str = f.read()
f.close()
generator = tf.keras.models.model_from_json(json_str, custom_objects=custom_objects)
generator.load_weights(join(log_dir, "model.h5"))


p = np.array(((2.1875, 0.0000,  0.0000,  0.0000),
            (0.0000, 2.1875,  0.0000,  0.0000),
            (0.0000, 0.0000, -1.8889, -2.3111),
            (0.0000, 0.0000, -1.0000,  0.0000)))

images = []

img_num = 100
for i in range(img_num):
    radius = 1.7
    shift = np.asarray((0.5, 0.1, 0.5)) #shift from voxelization
    center = np.array([0, 0.4, 0]) + shift

    angle = np.pi/2 - i * 2 * np.pi / img_num
    pos = np.array([radius * np.sin(angle), 0.5, radius * np.cos(angle)]) + shift

    mv = pyrr.matrix44.create_look_at(eye=pos, target=center, up=np.array([0, 1, 0])).transpose()

    matrix = np.matmul(p, mv)
    matrix = np.linalg.inv(matrix)

    tf.keras.backend.set_learning_phase(False)
    img = generator([model, matrix])
    img = img * 0.5 + 0.5
    img = tf.squeeze(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)

    images.append(np.array(img)) # for gif

    if not os.path.isdir(join(log_dir, name)):
        os.mkdir(join(log_dir, name))
    img = tf.image.encode_png(img)
    tf.write_file(join(log_dir, name, str(i).zfill(3)) + ".png", img)

if build_gif:
    imageio.mimsave(join(log_dir, name + ".gif"), images)



