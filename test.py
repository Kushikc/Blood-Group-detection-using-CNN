import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_original_cnn.h5")

# Check model summary
model.summary()
