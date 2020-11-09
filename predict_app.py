from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.models import Model
from tensorflow.keras.layers import (LSTM, Embedding, 
    TimeDistributed, Dense, RepeatVector, 
    Activation, Flatten, Reshape, concatenate,  
    Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import pickle
import requests
from flask import render_template,redirect
from flask import request
from flask import jsonify
from flask import Flask
from bs4 import BeautifulSoup
import random
import os
from jinja2 import Template

app = Flask(__name__)


def get_model():
    print('Start')
    global model
    ## TRY KERAS SAVE MODEL NEXT TIME
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.1)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(48,))
    se1 = Embedding(3663, 200, mask_zero=True)(inputs2)
    se2 = Dropout(0.1)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(3663, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
    caption_model.load_weights('caption-model.hdf5')
    print('* Model loaded *')
    model = caption_model
    global encode_model
    
    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)
    

def load_vocab():
    with open('vocab.txt', 'rb') as file:
        vocab = pickle.load(file)
    
    idxtoword = {}
    wordtoidx = {}
    ix = 1
    for w in vocab[:-1]:
        wordtoidx[w] = ix
        idxtoword[ix] = w
        ix += 1
    
    vocab_size = len(idxtoword)
    print(' * Vocabulary loaded * ')
    return idxtoword,wordtoidx,vocab_size

def generateCaption(photo):
    in_text = 'startseq'
    max_length = 48
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
    
def encode_image(image, method):
    WIDTH = 299
    HEIGHT = 299
    OUTPUT_DIM = 2048
    if method == 'GET':
        img = tensorflow.keras.preprocessing.image.load_img(image, target_size=(HEIGHT, WIDTH))
    else:
        img = image
    preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
    # Resize all images to a standard size (specified bythe image 
    # encoding network)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    # Convert a PIL image to a numpy array
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    # Expand to 2D array
    x = np.expand_dims(x, axis=0)
    # Perform any preprocessing needed by InceptionV3 or others
    x = preprocess_input(x)
    # Call InceptionV3 (or other) to extract the smaller feature set for 
    # the image.
    x = encode_model.predict(x) # Get the encoding vector for the image
    # Shape to correct form to be accepted by LSTM captioning network.
    x = np.array(x)
    x = np.reshape(x,OUTPUT_DIM)

    
    return x

print(' * Loading Models *')
get_model()
print('* Loading vocab *')
idxtoword,wordtoidx,vocab_size = load_vocab()


@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "GET":
        prediction_all = ""
        for i in range(20):
            img_path = f'img\{i}.jpg'
            processed_image = encode_image(img_path, 'GET')
            prediction = generateCaption(processed_image.reshape((1,2048)))
            prediction_all += prediction + '\n'
        return(prediction_all)

    if request.method == "POST":   
        message = request.get_json(force=True)
        print('message',message)
        if message['url']:
            response = requests.get('https:' + message['url'])
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')
            print('img open',image.mode)
            processed_image = encode_image(image, 'POST')
            print('img processed')
            prediction = generateCaption(processed_image.reshape((1,2048)))
            response = {
                'prediction': prediction    
            }
            print(prediction,jsonify(response))
            return jsonify(response)
        else:
            
            encoded = message['image']
            print('encode',len(encoded),encoded)
            decoded = base64.b64decode(encoded)
            print('decode')
            image = Image.open(io.BytesIO(decoded))
            image = image.convert('RGB')
            print('img open',image.mode)
            processed_image = encode_image(image, 'POST')
            print('img processed')
            prediction = generateCaption(processed_image.reshape((1,2048)))
            response = {
                'prediction': prediction    
            }
            print(prediction,jsonify(response))
            return jsonify(response)

@app.route('/wikiscrape/<article>',methods=['POST','GET'])


def scrape(article=''):
    S = requests.Session()
    
    URL = "https://en.wikipedia.org/w/api.php"
    if request.method == 'GET':
        page_title = article
    else:
        message = request.get_json(force=True)
        article = message['wiki_input']
        page_title = article
    PARAMS = {
            "action": "parse",
            "page": page_title,
            "format": "json"
        }
    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    page = (DATA["parse"]["text"]["*"])
        
    soup = BeautifulSoup(page, 'html.parser')
    thumb_divs = soup.findAll("div", {"class": "thumbinner"})

    images = []
    for div in thumb_divs:
        image = div.findAll("img")[0]['src']
        caption = div.findAll("div")[0].text

        image_and_caption = {
                'image_url' : image,
                'image_caption' : caption
            }
        images.append(image_and_caption)
    if request.method == 'GET':
        predictions = []
        img_url = []
        captions = []
        for img in images:
            url = 'https:'+img['image_url']
            captions.append(img['image_caption'])
            print(url)
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            img = img.convert('RGB')
            img_url.append(url)
            processed_image = encode_image(img,'POST')
            predictions.append(generateCaption(processed_image.reshape((1,2048))))
        print(predictions)
        return render_template('wikitemplate.html', img_url=img_url, predictions=predictions, captions=captions)
        
    else:
        return jsonify({'term' : page_title, 'images' : images })

 
    

@app.route('/generator',methods=['GET'])
def rand_generator():
    imgs = os.listdir('static/img')
    imgs = ['img/' + file for file in imgs]
    imgrand = random.sample(imgs,k=5)
    predictions = []
    for img in imgrand:
        processed_image = encode_image('static/'+img, 'GET')
        predictions.append(generateCaption(processed_image.reshape((1,2048))))
    return render_template('generator.html', imgrand=imgrand, predictions=predictions)
        


