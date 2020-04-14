import tensorflow as tf
import os
from models import ContentModel

content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file(
    'kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

EPOCHS = 5000
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

    style_layers = []

    return content_layers, style_layers


def content_loss(original, predicted):
    loss = []

    for val in range(len(original)):
        loss.append(tf.reduce_mean(tf.keras.losses.mse(original[val],
                                                       predicted[val])))

    return tf.reduce_mean(loss)

def clip_value(image):
    return tf.clip_by_value(
        clip_value_min = 0.,
        clp_value_max = 1.
    )

def main():
    content_image = load_image(content_path)
    print('Content Image Shape:', tf.shape(content_image))

    content_layers, _ = define_layers()
    content_extractor = ContentModel(content_layers)

    optimizer = tf.keras.optimizers.Adam()
    image = tf.Variable(tf.random.uniform(tf.shape(content_image),
                                          minval=0.,
                                          maxval=1.,
                                          dtype=tf.float32))

    target = content_extractor(content_image)

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            pred = content_extractor(image)
            loss = content_loss(target, pred)
        
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(clip_value(image))

        return loss
    
    for epoch in range(EPOCHS):
        loss = train_step(image)
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')
            save_img = tf.image.encode_jpeg(tf.cast(image[0] * 255, tf.uint8))
            tf.io.write_file(os.path.join(DIR, f'{epoch}.jpg'), save_img)
            print("Saved")

    