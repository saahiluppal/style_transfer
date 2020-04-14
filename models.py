import tensorflow as tf

class ContentModel(tf.keras.Model):
    def __init__(self, content_layers):
        super(ContentModel, self).__init__()
        vgg = tf.keras.applications.VGG19(
            include_top = False, weights = 'imagenet',
            pooling = 'avg'
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