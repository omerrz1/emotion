import pandas as pd
from datasets import load_dataset
from keras.utils import to_categorical
from nltk import PorterStemmer
import numpy as np
import keras
import pickle
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
    
dataset = load_dataset("go_emotions", "simplified")
data = pd.DataFrame(dataset['train'])
data = data.drop('id', axis='columns')

# processing labels 
data['labels'] = data['labels'].apply(lambda x: x[0])  # Extract the first element of each label
labels = to_categorical(data['labels'])

# processeing features
features = data['text']
stemmer = PorterStemmer()
stopwords = stopwords.words('english')
def clean_data(text):
    text = re.split('\W', text)
    text = [stemmer.stem(word) for word in text if word and word not in stopwords]
    cleaned_text = ' '.join(text)  # Join the cleaned words back into a single string
    return cleaned_text
print('processing data ....')
features = features.apply(clean_data)
print('tokenising data ....')
Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(features)
sequence = Tokenizer.texts_to_sequences(features)
vocabulary_size = len(Tokenizer.word_counts)+1
print('padding sequences ....')
max_len = 50
padded_sequence = pad_sequences(sequence,maxlen=max_len,padding='post')


# defining model

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=vocabulary_size,output_dim=100,input_length=max_len))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=28,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

config = {
    'epochs':100,
    'batch_size':15,
}
model.fit(padded_sequence,labels,**config,verbose=1)
model.save('emotins.h5')
pickle.dump(Tokenizer,open('emotions_tokeniser.pickle','wb'))
