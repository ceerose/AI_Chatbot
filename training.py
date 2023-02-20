import random  # for choosing a random response at the end
import json
import pickle  # for serialization
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer # reduces the word to its stem, so that we don't lose any performance b/c it's looking for an exact word

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow. keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read()) # reading the json file as text, passing it to the load function

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ','] # letters that won't be taken into account'

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatizer(word) for words if word is not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#machine learning part; representing words as numerical values
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# building neural network model
# adding different layers
model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax')) # activation that sums up (scales) the results in the ouput layer so that they add up to one

sgd = SGD(lr=0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimixer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.model.h5', hist)
print("Done")
