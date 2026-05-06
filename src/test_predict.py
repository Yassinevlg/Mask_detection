import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image

def test_model():
    print("Downloading image...")
    # Get a face with a mask
    mask_url = "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/data/with_mask/0-with-mask.jpg"
    nomask_url = "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/data/without_mask/0.jpg"
    
    mask_path = tf.keras.utils.get_file("mask.jpg", mask_url)
    nomask_path = tf.keras.utils.get_file("nomask.jpg", nomask_url)
    
    model = tf.keras.models.load_model('models/mask_detection_model.keras')
    
    def predict(path):
        img = Image.open(path).convert('RGB').resize((128, 128))
        arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
        return model.predict(arr[None, ...], verbose=0)[0]

    print(f"Prediction for WITH MASK: {predict(mask_path)}")
    print(f"Prediction for WITHOUT MASK: {predict(nomask_path)}")

test_model()
