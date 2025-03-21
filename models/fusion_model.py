import tensorflow as tf

class HybridPredictiveModel(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        # Define layers with input_dim for dynamic input shaping
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
