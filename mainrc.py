import tensorflow as tf
from models import ContentModel, StyleModel
import os
DIR = './save'

content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file(
    'kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def load_image(path):
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


content_layers = ['block4_conv2',
                  'block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_style_layers = len(style_layers)
num_content_layers = len(content_layers)

content_image = load_image(content_path)
style_image = load_image(style_path)

style = StyleModel(style_layers)
content = ContentModel(content_layers)

style_target = style(style_image)
content_target = content(content_image)


def content_loss(original, predicted):
    loss = []

    for val in range(len(original)):
        loss.append(tf.reduce_mean(tf.keras.losses.mse(original[val],
                                                       predicted[val])))

    loss = (1.0 / float(len(content_layers))) * tf.reduce_mean(loss)
    return loss


image = tf.Variable(content_image)


def style_loss(original, predicted):
    loss = []

    for val in range(len(original)):
        loss.append(tf.reduce_mean(tf.keras.losses.mse(original[val],
                                                       predicted[val])))

    loss = (1.0 / float(len(style_layers))) * tf.reduce_mean(loss)
    return loss


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)


def calculate_loss(style_output, content_output, image):
    s = 0.75 * style_loss(style_target, style_output)
    c = 0.25 * content_loss(content_target, content_output)
    v = 30 * tf.image.total_variation(image)
    return s + c + v


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        style_output = style(image)
        content_output = content(image)
        loss = calculate_loss(style_output, content_output, image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


for epoch in range(1000):
    train_step(image)

save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
tf.io.write_file(os.path.join(DIR, '1.jpg'), save_img)

for epoch in range(1000):
    train_step(image)

save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
tf.io.write_file(os.path.join(DIR, '2.jpg'), save_img)

for epoch in range(1000):
    train_step(image)

save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
tf.io.write_file(os.path.join(DIR, '3.jpg'), save_img)
