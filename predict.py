import tensorflow as tf 
# from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
from PIL import Image

classifierLoad = tf.keras.models.load_model('model.h5') 

import numpy as np

test_image = Image.open('ISIC_0052060.jpg')
test_image = test_image.resize((200,200))
test_image = np.array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifierLoad.predict(test_image)
predicted_class = np.argmax(result[0])

if predicted_class == 0:
    print("Below 10% .... Its beginning stage")
elif predicted_class == 1:
    print("Below 50% .... You may consult the doctor")
elif predicted_class == 2:
    print("Below 80% .... Take medicine regularly")
elif predicted_class == 3:
    print("Above 80% .... You are at risk, it's final stage - Surgery Needed")