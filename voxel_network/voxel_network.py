import datetime
import os, math
from os.path import join

import numpy as np
import pyrr
import tensorflow as tf

import custom_layers

tf.enable_eager_execution()

IMG_WIDTH = 256
IMG_HEIGHT = 256

VOXEL_NUM = 32
POINTS_PER_VOXEL = 32
CHANNELS = 9

# how many of the images are used for training, max 1
TRAIN_FRACTION = 0.9

NUM_EPOCHS = 100

current_dir = os.path.dirname(__file__)
model_file = join(current_dir, "01.npy")
image_dir = join(current_dir, "data", "sketch")
log_dir = join(current_dir, "log")


def load_model(file):
    model = np.load(file)
    model = tf.convert_to_tensor(model, dtype=tf.float32)
    model = tf.reshape(model, [1, VOXEL_NUM, VOXEL_NUM, VOXEL_NUM, POINTS_PER_VOXEL, CHANNELS])
    return model


def load_image(image_file):
    image = tf.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_images(image, size=[IMG_HEIGHT, IMG_WIDTH],
                                   align_corners=True)
    # normalizing the images to [-1, 1]
    image = (image / 127.5) - 1

    return image

def save_img(img, file):
    dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(dir):
        os.mkdir(dir)

    img = img * 0.5 + 0.5
    img = tf.squeeze(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
    img = tf.image.encode_png(img)
    tf.write_file(file, img)


#########################################
# Generator
#########################################


def my_generator():
    def conv3d(input_tensor, num_filters):
        initializer = tf.random_normal_initializer(0., 0.02)
        tensor = tf.keras.layers.Conv3D(num_filters, (4, 4, 4), strides=2, padding='same',
                                        kernel_initializer=initializer)(input_tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.LeakyReLU(0.2)(tensor)
        return tensor

    def deconv3d(input_tensor, skip_tensor, num_filters):
        initializer = tf.random_normal_initializer(0., 0.02)
        tensor = tf.keras.layers.Conv3DTranspose(num_filters, (4, 4, 4), strides=2, padding='same',
                                                 kernel_initializer=initializer)(input_tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Activation('relu')(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1)([tensor, skip_tensor])
        return tensor

    def conv(input_tensor, num_filters, batchnorm=True):
        tensor = tf.keras.layers.Conv2D(num_filters, (4, 4), strides=2,
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))(input_tensor)
        if batchnorm:
            tensor = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.8,
                                                        gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(
                tensor)
        tensor = tf.keras.layers.Dropout(0.2)(tensor)
        tensor = tf.keras.layers.LeakyReLU(0.2)(tensor)
        return tensor

    def deconv(input_tensor, skip_tensor, num_filters, dropout=False):
        tensor = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2,
                                                 padding='same',
                                                 kernel_initializer=tf.random_normal_initializer(0., 0.02))(
            input_tensor)
        tensor = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.8,
                                                    gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

        tensor = tf.keras.layers.Dropout(0.2)(tensor)
        tensor = tf.keras.layers.Activation('relu')(tensor)
        if skip_tensor != None:
            tensor = tf.keras.layers.Concatenate(axis=-1)([tensor, skip_tensor])
        return tensor

    input = tf.keras.layers.Input(shape=(VOXEL_NUM, VOXEL_NUM, VOXEL_NUM, POINTS_PER_VOXEL, CHANNELS))
    input_matrix = tf.keras.layers.Input(shape=(4, 4))

    tensor = tf.keras.layers.Reshape([VOXEL_NUM, VOXEL_NUM, VOXEL_NUM, POINTS_PER_VOXEL * CHANNELS])(input)

    tensor = tf.keras.layers.Conv3D(256, 4, strides=1, padding='same')(tensor)
    tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.layers.LeakyReLU(0.2)(tensor)

    tensor = tf.keras.layers.Conv3D(128, 4, strides=1, padding='same')(tensor)
    tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.layers.LeakyReLU(0.2)(tensor)

    inpainting1 = conv3d(tensor, 64)
    inpainting2 = conv3d(inpainting1, 128)
    inpainting3 = conv3d(inpainting2, 256)

    inpainting4 = deconv3d(inpainting3, inpainting2, 128)
    inpainting5 = deconv3d(inpainting4, inpainting1, 64)
    inpainting6 = deconv3d(inpainting5, tensor, 32)

    projection = custom_layers.ProjectionLayer(reduce_dim=False)([inpainting6, input_matrix])

    #add distance-to-cam info to channels
    projection_with_depth = custom_layers.ConcatDepth(num_layers=1)(projection)

    # same size compression
    occ1 = tf.keras.layers.Conv3D(4, 4, strides=1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(0., 0.02))(projection_with_depth)
    occ1 = tf.keras.layers.BatchNormalization()(occ1)
    occ1 = tf.keras.layers.LeakyReLU(0.2)(occ1) # 32

    # rest of occlusion network
    occ2 = conv3d(occ1, 4) # 16
    occ3 = conv3d(occ2, 8) # 8
    occ4 = conv3d(occ3, 16) # 4

    occ5 = deconv3d(occ4, occ3, 16) # 8
    occ6 = deconv3d(occ5, occ2, 8) # 16
    occ7 = deconv3d(occ6, occ1, 4) # 32

    occ_out = tf.keras.layers.Conv3D(1, 1, padding='same')(occ7)
    occ_out = tf.keras.layers.BatchNormalization()(occ_out)
    occ_out = tf.keras.layers.Softmax(axis=1)(occ_out)

    weighted_projection = tf.keras.layers.Multiply()([projection, occ_out])
    reduced_projection = custom_layers.ReduceSum(axis=1)(weighted_projection)

    decoder1 = conv(reduced_projection, 256)
    decoder2 = conv(decoder1, 512)
    decoder3 = conv(decoder2, 512)

    decoder4 = deconv(decoder3, decoder2, 512)
    decoder5 = deconv(decoder4, decoder1, 512)
    decoder6 = deconv(decoder5, reduced_projection, 512)
    decoder = deconv(decoder6, None, 256)
    decoder = deconv(decoder, None, 128)
    decoder = deconv(decoder, None, 64)

    output = tf.keras.layers.Conv2D(3, (4, 4), padding='same',
                                    kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                    activation='tanh')(decoder)

    return tf.keras.Model(inputs=[input, input_matrix], outputs=[output])


#######################################
# Losses
#######################################


def image_loss(generated, target):
    l1_loss = tf.reduce_mean(tf.abs(generated - target))
    return l1_loss


def validation_loss(input, target):
    return tf.reduce_mean(tf.abs(target - input))


#########################################
# Training
######################################


model = load_model(model_file)
model_name = os.path.splitext(os.path.basename(model_file))

images = [join(image_dir, file) for file in sorted(os.listdir(image_dir)) if file.startswith(model_name)]
num_images = len(images)
num_train = math.floor(TRAIN_FRACTION * num_images)
print("Found " + str(num_images) + " images for this model")

# create matrices according to render convention
matrices = []

p = np.array(((2.1875, 0.0000,  0.0000,  0.0000),
            (0.0000, 2.1875,  0.0000,  0.0000),
            (0.0000, 0.0000, -1.8889, -2.3111),
            (0.0000, 0.0000, -1.0000,  0.0000)))

r = 1.7
shift = np.asarray((0.5, 0.1, 0.5))
center = np.array([0. ,0.4 , 0.]) + shift
for i in range(len(images)):
    np.random.seed(i)
    theta = np.random.rand() * np.pi
    a = np.random.rand() * 2 - 1
    phi = np.arccos(a)
    x = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.sin(phi)
    y = r * np.cos(phi)
    pos = np.array([x, z, -y]) + shift
    mv = pyrr.matrix44.create_look_at(eye=pos, target=center, up=np.array([0, 1, 0])).transpose()

    matrix = np.matmul(p, mv)
    matrix = np.linalg.inv(matrix)
    matrices.append(matrix.tolist())

images = tf.data.Dataset.from_tensor_slices(images)
matrices = tf.data.Dataset.from_tensor_slices(matrices)
dataset = tf.data.Dataset.zip((images, matrices)).shuffle(100, seed=42)
train_dataset, test_dataset = dataset.take(num_train), dataset.skip(num_train)

train_dataset = train_dataset.shuffle(200, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(lambda x, y: (load_image(x), y))
train_dataset = train_dataset.batch(1)

test_dataset = test_dataset.shuffle(200, reshuffle_each_iteration=True)
test_dataset = test_dataset.map(lambda x, y: (load_image(x), y))
test_dataset = test_dataset.batch(1)

generator = my_generator()
generator.summary(line_length=200)
#tf.keras.utils.plot_model(generator, to_file='model.png', show_shapes=True)

optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
global_step = tf.train.get_or_create_global_step()

log_subdir = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
writer = tf.contrib.summary.create_file_writer(join(log_dir, log_subdir), flush_millis=10000)

for epoch in range(NUM_EPOCHS):
    print("Epoch {}".format(epoch))

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        gen_loss_avg = tf.contrib.eager.metrics.Mean()

        # tensorboard stuff
        for image, matrix in test_dataset.take(1):
            generated = generator([model, matrix])
            tf.contrib.summary.image("generated", generated)
            tf.contrib.summary.image("target", image)

            compared = tf.concat([generated, image], axis=2)
            save_img(compared, join(log_dir, log_subdir, "img_target", str(epoch).zfill(3) + ".png"))
            save_img(generated, join(log_dir, log_subdir, "img", str(epoch).zfill(3) + ".png"))

        # log weights
        for i, w in enumerate(generator.trainable_variables):
            tf.contrib.summary.histogram("gen" + str(i) + w.name, w)

        ########
        # train
        ########
        tf.keras.backend.set_learning_phase(True)
        for image, matrix in train_dataset:

            with tf.GradientTape() as tape:
                generated = generator([model, matrix])
                loss = image_loss(generated, image)

            gradients = tape.gradient(loss, generator.variables)
            optimizer.apply_gradients(zip(gradients, generator.variables), global_step=global_step)

            gen_loss_avg(loss)

        tf.contrib.summary.scalar("generator loss", gen_loss_avg.result())

        # validation
        tf.keras.backend.set_learning_phase(False)
        val_loss_avg = tf.contrib.eager.metrics.Mean()
        for image, matrix in test_dataset:
            generated = generator([model, matrix])
            loss_value = validation_loss(generated, image)
            val_loss_avg(loss_value)
        tf.contrib.summary.scalar("validation loss", val_loss_avg.result())

        print(gen_loss_avg.result(), val_loss_avg.result())

    # Architecture
    with open(os.path.join(log_dir, log_subdir, "model.json"), "w") as f:
        f.write(generator.to_json())
    f.close()
    # Weights
    generator.save_weights(os.path.join(log_dir, log_subdir, "model.h5"))


