import tensorflow as tf
import os
from models import ContentModel, StyleModel

content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file(
    'kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

EPOCHS = 10000
DIR = 'saved'

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


def define_layers():
    content_layers = ['block4_conv2',
                      'block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    return content_layers, style_layers


def content_loss(original, predicted, len_content_layers):
    loss = []

    for val in range(len(original)):
        loss.append(tf.reduce_mean(tf.keras.losses.mse(original[val],
                                                       predicted[val])))

    loss = (1.0 / float(len_content_layers)) * tf.reduce_mean(loss)
    return loss


def style_loss(original, predicted, len_style_layers):
    loss = []

    for val in range(len(original)):
        loss.append(tf.reduce_mean(tf.keras.losses.mse(original[val],
                                                       predicted[val])))

    loss = (1.0 / float(len_style_layers)) * tf.reduce_mean(loss)
    return loss

def calculate_loss(style_output, style_target, content_output,
                        content_target, len_style, len_content,
                        image):
    s = 0.75 * style_loss(style_target, style_output, len_style)
    c = 0.25 * content_loss(content_target, content_output, len_content)
    v = 30 * tf.image.total_variation(image)
    return s + c + v

def clip_value(image):
    return tf.clip_by_value(
        image,
        clip_value_min=0.,
        clip_value_max=1.
    )


def main():
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    print('Content Image Shape:', tf.shape(content_image))

    content_layers, style_layers = define_layers()
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    content_extractor = ContentModel(content_layers)
    style_extractor = StyleModel(style_layers)

    optimizer = tf.keras.optimizers.Adam(0.02)
    image = tf.Variable(tf.random.uniform(tf.shape(content_image),
                                          minval=0.,
                                          maxval=1.,
                                          dtype=tf.float32))

    content_target = content_extractor(content_image)
    style_target = style_extractor(style_image)

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            style_output = style_extractor(image)
            content_output = content_extractor(image)
            loss = calculate_loss(style_output, style_target,
                                content_output, content_target,
                                num_style_layers, num_content_layers,
                                image)

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(clip_value(image))

        return loss

    print('Start...')
    for epoch in range(EPOCHS):
        loss = train_step(image)
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')
            save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
            tf.io.write_file(os.path.join(DIR, f'{epoch}.jpg'), save_img)
            print("Saved")
    print('End...')
    save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
    tf.io.write_file(os.path.join(DIR, f'{epoch}.jpg'), save_img)
    print('Saved...')


main()
