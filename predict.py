#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from flask import Flask
from flask import request
from flask import jsonify
import base64
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import itertools
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras. metrics import categorical_crossentropy
from random import randint

from scipy import ndimage
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import backend as K
import PIL
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import io


app=Flask(__name__)


# def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
#  if type(ims[0]) is np.ndarray:
#   ims = np.array(ims).astype(np.uint8)
#   if(ims.shape[-1]!=3):
#    ims=ims.transpose((0,2,3,1))
#  f=plt.figure(figsize=figsize)
#  cols = len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows+1
#  for i in range(len(ims)):
#   sp = f.add_subplot(rows,cols,i+1)
#   sp.axis('Off')
#   if titles is not None:
#    sp.set_title(titles[i],fontsize=16)
#   plt.imshow(ims[i],interpolation=None if interp else 'none')
#  plt.show()


model=load_model("model_after_vgg.h5")
print('Model loaded')
#print(model.summary())
def preprocess(image,target_size):
	image= image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image,axis=0)
	return image
@app.route('/predict', methods=['POST'])
def predict():
	message=request.get_json(force=True)
	encoded_img=message['img']
	decoded_img=base64.b64decode(encoded_img)
	image = Image.open(io.BytesIO(decoded_img))
	processed_img = preprocess(image,target_size=(224,224))
	prediction = model.predict(processed_img)
	#print(prediction)
	name=''
	index = prediction[0].argmax()
	if  index == 0:
		name="MOM"
	elif index==1:
		name="NANU"
	elif index ==2:
		name="Prabesh"
	else:
		name="Sanu"
	confidence=np.max(prediction[0])*100
	toSend = {'prediction':name,'confidence':confidence}
	print(toSend, file=sys.stdout)
	return jsonify(toSend)
	# print(response, file=sys.stderr)






