import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflowjs as tfjs

IMAGE_SIZE = 128
model = tf.keras.models.load_model('models/best_model.keras')

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights=None
)
inf_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
])

# Copy weights
inf_model.layers[0].set_weights(model.layers[2].get_weights()) # MobileNetV2
inf_model.layers[1].set_weights(model.layers[3].get_weights()) # GlobalAvg
inf_model.layers[2].set_weights(model.layers[5].get_weights()) # Dense(64)
inf_model.layers[3].set_weights(model.layers[7].get_weights()) # Dense(2)

print("Saving to tfjs...")
tfjs.converters.save_keras_model(inf_model, 'models/tfjs_model')
print("Done!")
