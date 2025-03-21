# models/test_conversion.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must be before TensorFlow import

import tensorflow as tf

# Create a minimal test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

# Test conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/test_model.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("TFLite conversion successful!")
