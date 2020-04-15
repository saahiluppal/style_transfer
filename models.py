import tensorflow as tf


class StyleModel(tf.keras.Model):
    def __init__(self, style_layers):
        super(StyleModel, self).__init__()
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet',
            pooling='avg'
        )
        outputs = [vgg.get_layer(name).output
                   for name in style_layers]

        self.model = tf.keras.Model(vgg.input, outputs)
        self.model.trainable = False

    def gram_matrix(self, tensor):
        channels = tf.shape(tensor)[-1]
        features = tf.reshape(tensor, (-1, channels))
        norm = tf.shape(features)[0]

        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(norm, tf.float32)

    def call(self, x):
        x = x * 255
        x = tf.keras.applications.vgg19.preprocess_input(x)
        output = self.model(x)
        gm = [self.gram_matrix(val) for val in output]
        return gm


class ContentModel(tf.keras.Model):
    def __init__(self, content_layers):
        super(ContentModel, self).__init__()
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet',
            pooling='avg'
        )
        outputs = [vgg.get_layer(name).output
                   for name in content_layers]

        self.model = tf.keras.Model(vgg.input, outputs)
        self.model.trainable = False

    def call(self, x):
        x = x * 255
        x = tf.keras.applications.vgg19.preprocess_input(x)
        output = self.model(x)
        return output
