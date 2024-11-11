import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Embedding, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk

"""
Running in Google Collab was the best way for me to resolve any package errors
"""

df = pd.read_csv('data.csv')

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwordss = stopwords.words('english')
lem = WordNetLemmatizer()

def clean(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    words = [word for word in line.split() if word not in stopwordss]
    return ' '.join(words)

df['Sentence'] = df['Sentence'].apply(clean)
df['Sentiment'].isnull().sum()

le = LabelEncoder()
df['status'] = le.fit_transform(df['Sentiment'])
x = df['Sentence']
y = pd.get_dummies(df['Sentiment']).values

tokenizer = Tokenizer(oov_token='<unk>', num_words=2500)
tokenizer.fit_on_texts(x.values)
data_x = tokenizer.texts_to_sequences(x.values)
vocab = tokenizer.word_index
l_voc = len(vocab)

pad_sz = 42
data_x = pad_sequences(data_x, maxlen=pad_sz, padding='post', truncating='post')

x_train, x_test, y_train, y_test = train_test_split(data_x, y, random_state=0, shuffle=True, stratify=y, test_size=0.2)

configurations = [
    {'em_sz': 100, 'latent_sz': 256, 'dropout_rate': 0.2, 'filters': 64, 'kernel_size': 3, 'batch_size': 128},
    {'em_sz': 50, 'latent_sz': 200, 'dropout_rate': 0.3, 'filters': 128, 'kernel_size': 5, 'batch_size': 64},
    {'em_sz': 75, 'latent_sz': 128, 'dropout_rate': 0.25, 'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'batch_size': 128}
]

for i, config in enumerate(configurations):
    print(f"\n--- Training Model Configuration {i+1} ---\n")

    model = Sequential()
    model.add(Embedding(l_voc, config['em_sz'], input_length=pad_sz))
    model.add(Dropout(config['dropout_rate']))
    model.add(Conv1D(config['filters'], config['kernel_size'], padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=config.get('pool_size', 4)))  # Default to 4 if not specified
    model.add(LSTM(config['latent_sz']))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=10, validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test)
    print(f"Configuration {i+1} - Test Loss: {score}, Test Accuracy: {acc}")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy - Config {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
