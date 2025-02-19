import tensorflow as tf

# Load the trained .keras model
model = tf.keras.models.load_model("/Users/apple/Downloads/Leukemia_project/leukemia_model.h5", compile=False)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize model size
tflite_model = converter.convert()

# Save the converted model
with open("model/leukemia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TensorFlow Lite format!")
