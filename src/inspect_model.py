import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Loading best_model.keras...")
model = tf.keras.models.load_model('models/best_model.keras')
model.summary()
