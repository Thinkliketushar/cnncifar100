#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os
import tensorflow 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

#load model
labels =  ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
           'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
           'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 
           'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 
           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
           'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
           'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
           'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
           'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
labels_names = []
for i in range(len(labels)):
  labels_names += [i]
reverse_mapping = dict(zip(labels_names, labels)) 
def mapper(value):
  return reverse_mapping[value]

def pred_class100(imagefromdb):
  test_image = load_img(imagefromdb, target_size = (32, 32)) 
  print("Got Image for prediction")
  test_image = img_to_array(test_image)/255 
  test_image = np.expand_dims(test_image, axis = 0)
  model =load_model("ass2.h5")
  mresult = model.predict(test_image).round(3)
  aresult = np.argmax(mresult)
  result = mapper(aresult)
  print(f'Prediction is {result}.')
  return result

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] 
        filename = file.filename        
        print("Input Image File Name = ", filename)
        file_path = os.path.join('static/user uploaded/', filename)
        file.save(file_path)
        print("Input Image File Path = ", file_path)
        print("Let's Predicting Image Name")
        pred = pred_class100(imagefromdb=file_path)
        return render_template('predict.html', pred = pred, user_image = file_path)
    
if __name__ == "__main__":
    app.run(debug=True) 
    
    
    
    
    
    
    
    