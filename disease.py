import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
model = keras.models.load_model('bestmodel.h5')
class_names=['BUMPS','HAIR LOSS','HOT SPOTS','RASHES','SORES']
upload_image = st.file_uploader(label='Upload image', type=['png', 'jpg','jpeg'], accept_multiple_files=False)
if upload_image is not None:
	image = Image.open(upload_image)
	converted_img = np.array(image.convert('RGB'))
	img = cv2.resize(converted_img, dsize=(128, 128))
	img_reshape = np.reshape(img,[1,128,128,3])
	y_predict = class_names[np.argmax(model.predict(img_reshape), axis=1)[0]]
	st.text('Disease : ' + y_predict)