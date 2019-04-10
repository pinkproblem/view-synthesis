import datetime
import os
from os.path import join

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

IMG_WIDTH = 256
IMG_HEIGHT = 256

BATCH_SIZE = 1
NUM_EPOCHS = 100

current_dir = os.path.dirname(__file__)
input_dir = join(current_dir, "train")
test_dir = join(current_dir, "test")
depth_dir = join(current_dir, "data", "depth")
normal_dir = join(current_dir, "data", "normal")
target_dir = join(current_dir, "data", "target")
target_depth_dir = join(current_dir, "data", "target_depth")
log_dir = join(current_dir, "log")


def load_images(image_file, depth_file, normal_file, target_file, target_depth_file):
    def load_png(file, gray=False, hue=False):
        image = tf.read_file(file)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=[IMG_HEIGHT, IMG_WIDTH], align_corners=True)
        if gray:
            image = tf.image.rgb_to_grayscale(image)
        elif hue:
            image = tf.image.rgb_to_hsv(image)
            image = tf.slice(image, [0, 0, 0], [256, 256, 1])
        image = (image * 2) - 1
        return image

    input_image = load_png(image_file)
    depth_image = load_png(depth_file, gray=True)
    normal_image = load_png(normal_file, hue=True)
    target_image = load_png(target_file, gray=True)
    target_depth_image = load_png(target_depth_file, gray=True)

    return (input_image, depth_image, normal_image, target_image, target_depth_image)


#########################################
# Generator
#########################################

def my_generator():
    def conv(input_tensor, num_filters, batchnorm=True):
        tensor = tf.keras.layers.Conv2D(num_filters, (4, 4), strides=2,
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))(input_tensor)
        if batchnorm:
            tensor = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.8,
                                                        gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(
                tensor, training=True)
        tensor = tf.keras.layers.LeakyReLU(0.2)(tensor)
        return tensor

    def deconv(input_tensor, skip_tensor, num_filters, dropout=False):
        tensor = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2,
                                                 padding='same',
                                                 kernel_initializer=tf.random_normal_initializer(0., 0.02))(
            input_tensor)
        tensor = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.8,
                                                    gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor,
                                                                                                               training=True)
        if dropout:
            tensor = tf.keras.layers.Dropout(0.5)(tensor, training=True)
        tensor = tf.keras.layers.Activation('relu')(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1)([tensor, skip_tensor])
        return tensor

    input = tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 5))

    encoder1 = conv(input, 64, batchnorm=False)
    encoder2 = conv(encoder1, 128)
    encoder3 = conv(encoder2, 256)
    encoder4 = conv(encoder3, 512)
    encoder5 = conv(encoder4, 512)
    encoder6 = conv(encoder5, 512)
    encoder7 = conv(encoder6, 512)
    encoder8 = conv(encoder7, 512)

    decoder1 = deconv(encoder8, encoder7, 512, dropout=True)
    decoder2 = deconv(decoder1, encoder6, 1024, dropout=True)
    decoder3 = deconv(decoder2, encoder5, 1024, dropout=True)
    decoder4 = deconv(decoder3, encoder4, 1024)
    decoder5 = deconv(decoder4, encoder3, 512)
    decoder6 = deconv(decoder5, encoder2, 256)
    decoder7 = deconv(decoder6, encoder1, 128)

    depth_decoder1 = deconv(encoder8, encoder7, 512, dropout=True)
    depth_decoder2 = deconv(depth_decoder1, encoder6, 512, dropout=True)
    depth_decoder3 = deconv(depth_decoder2, encoder5, 512, dropout=True)
    depth_decoder4 = deconv(depth_decoder3, encoder4, 512)
    depth_decoder5 = deconv(depth_decoder4, encoder3, 512)
    depth_decoder6 = deconv(depth_decoder5, encoder2, 256)
    depth_decoder7 = deconv(depth_decoder6, encoder1, 128)

    output = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=2,
                                             padding='same',
                                             kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                             activation='tanh')(decoder7)

    depth_output = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=2,
                                                   padding='same',
                                                   kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                   activation='tanh')(depth_decoder7)

    return tf.keras.Model(inputs=[input], outputs=[output, depth_output])


#######################################
# Losses
#######################################


def image_loss(generated, target, depth_out, depth_real):
    l1_loss = tf.reduce_mean(tf.abs(generated - target))
    aux_loss = tf.reduce_mean(tf.abs(depth_out - depth_real))
    return l1_loss + aux_loss


def validation_loss(input, target):
    return tf.reduce_mean(tf.abs(target - input))


#########################################
# Training
######################################


input = [os.path.join(input_dir, file) for file in sorted(os.listdir(input_dir))]

basenames = [os.path.splitext(os.path.basename(file))[0] for file in input]
depth = [os.path.join(depth_dir, file + ".png") for file in basenames]
normal = [os.path.join(normal_dir, file + ".png") for file in basenames]
target = [os.path.join(target_dir, file + ".png") for file in basenames]
target_depth = [os.path.join(target_depth_dir, file + ".png") for file in basenames]

train_dataset = list(zip(input, depth, normal, target, target_depth))
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(500, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(lambda x: load_images(x[0], x[1], x[2], x[3], x[4]))
train_dataset = train_dataset.batch(1)



input = [os.path.join(test_dir, file) for file in sorted(os.listdir(test_dir))]

basenames = [os.path.splitext(os.path.basename(file))[0] for file in input]
depth = [os.path.join(depth_dir, file + ".png") for file in basenames]
normal = [os.path.join(normal_dir, file + ".png") for file in basenames]
target = [os.path.join(target_dir, file + ".png") for file in basenames]
target_depth = [os.path.join(target_depth_dir, file + ".png") for file in basenames]

test_dataset = list(zip(input, depth, normal, target, target_depth))
test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
test_dataset = test_dataset.map(lambda x: load_images(x[0], x[1], x[2], x[3], x[4]))
test_dataset = test_dataset.batch(1)

generator = my_generator()
generator.summary()
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
        for input, depth, normal, target, target_depth in train_dataset.take(1):
            generated, generated_depth = generator(tf.concat([input, depth, normal], axis=-1))
            tf.contrib.summary.image("train_input", input)
            tf.contrib.summary.image("train_depth", depth)
            tf.contrib.summary.image("train_normal", normal)
            tf.contrib.summary.image("train_generated", generated)
            tf.contrib.summary.image("train_gen_depth", generated_depth)
            tf.contrib.summary.image("train_target", target)

        # log weights
        for i, w in enumerate(generator.trainable_variables):
            tf.contrib.summary.histogram("gen" + str(i) + w.name, w)

        ########
        # train
        ########
        tf.keras.backend.set_learning_phase(True)
        for input, depth, normal, target, target_depth in train_dataset:
            with tf.GradientTape() as img_tape:
                generated, generated_depth = generator(tf.concat([input, depth, normal], axis=-1))
                loss = image_loss(generated, target, generated_depth, target_depth)

            img_gradients = img_tape.gradient(loss, generator.variables)
            optimizer.apply_gradients(zip(img_gradients, generator.variables))

            # log actual per image loss
            gen_loss_avg(image_loss(generated, target, generated_depth, target_depth))

        tf.contrib.summary.scalar("generator loss", gen_loss_avg.result())


        # validation loss
        tf.keras.backend.set_learning_phase(False)
        val_loss_avg = tf.contrib.eager.metrics.Mean()
        for input, depth, normal, target, target_depth in test_dataset:
            generated, generated_depth = generator(tf.concat([input, depth, normal], axis=-1))
            loss_value = validation_loss(generated, target)
            val_loss_avg(loss_value)
        tf.contrib.summary.scalar("validation loss", val_loss_avg.result())

    # save model
    with open(os.path.join(log_dir, log_subdir, "model.json"), "w") as f:
        f.write(generator.to_json())
    f.close()
    generator.save_weights(os.path.join(log_dir, log_subdir, "model.h5"))