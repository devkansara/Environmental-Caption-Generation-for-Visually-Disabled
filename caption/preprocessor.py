import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
from IPython import display

def get_model():
    #BASE_DIR = '/Users/dev/Desktop/Image Recommendation System/working_directory/Images/'
    WORKING_DIR = '/Users/dev/Desktop/Caption Recommendation System/working_directory/Files'
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    max_length=35
    vocab_size=8485

    # encoder model
    # image feature layers
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.2)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    model.load_weights('/Users/dev/Desktop/Caption Recommendation System/working_directory/Files/optimized_model_20.h5')
    return model

def predict_caption(model, image, tokenizer, max_length):
    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def predict_new_image(uploaded_file):
    max_length=35
    tokenizer=pickle.load(open('/Users/dev/Desktop/Caption Recommendation System/working_directory/Files/tokenizer.pkl', 'rb'))
    #image = load_img(uploaded_file, target_size=(224, 224))
    image = Image.open(uploaded_file)
    image = image.resize((224,224))
    plt.imshow(image)
    image = img_to_array(image) #Converting Image
    image = image.reshape((1, 224, 224, 3)) # Reshaping Data for Model
    image = preprocess_input(image) #Preprocessing Images
    model_init = VGG16()
    model_init = Model(inputs=model_init.inputs, outputs=model_init.layers[-2].output)
    feature = model_init.predict(image, verbose=0)
    model = get_model()
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    return y_pred[8:-7]

def audio(uploaded_file):
    language = 'en'
  
# Passing the text and language to the engine, 
    myobj = gTTS(text=predict_new_image(uploaded_file), lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
    audio_file_name = "Predicted_text.mp3"
  
# Playing the converted file
    myobj.save(audio_file_name)
    return audio_file_name