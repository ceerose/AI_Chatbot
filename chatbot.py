import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model()

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

# functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# converts sentences into a bag of words; i.e., lists of 0s and 1s that indicate if the word is there
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# defining predict function
def predict_class(sentence):
    bow = bag_of_words(sentence) # what we need to feed into the neural network
    res = model.predict(np.array([bow]))[0] #gets our prediction
    ERROR_THRESHOLD = 0.25 # allows for certain uncertainties but if uncertainty is too high we won't take it into results
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# we create a bag of worms
# predict the result based on those bag of worms
# then we have a certain threshold so we don't have too much uncertainty
# we enumerate the results so that we get the index/class and probability
# then we sort by probability in reverse order so we have the highest probaility first
# and then have a return list full of classes and probabilities

