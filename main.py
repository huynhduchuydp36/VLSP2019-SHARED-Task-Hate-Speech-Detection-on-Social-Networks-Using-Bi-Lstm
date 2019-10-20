import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model




EMBEDDING_FILE = './Word Embedding/cc.vi.300.vec'
train_x = pd.read_csv('./Data/Train.csv').fillna(" ")
test_x = pd.read_csv('./Data/Test.csv').fillna(" ")



max_features=7000
maxlen=150
embed_size=300

train_x['free_text'].fillna(' ')

test_x['free_text'].fillna(' ')
train_y = train_x[['CLEAN', 'OFFENSIVE', 'HATE']].values

train_x = train_x['free_text'].str.lower()

test_x = test_x['free_text'].str.lower()
# Vectorize text + Prepare  Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
print("create vector")
embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

out = Dense(3, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prediction
batch_size = 32
epochs = 3

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

print(predictions)
result = pd.read_csv('./Data/Test.csv')
result[['CLEAN', 'OFFENSIVE', 'HATE']] = predictions
# submission.to_csv('Thunghiem1.csv', index=False)
model_json = model.to_json()
with open("model_num_bc.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
for i in range(len(result)):
    if (result['CLEAN'][i] >= result['OFFENSIVE'][i] and result['CLEAN'][i]>=result['HATE'][i]):
        result['label_id'][i]=int(0)
    elif(result['OFFENSIVE'][i] >= result['CLEAN'][i] and result['OFFENSIVE'][i]>=result['HATE'][i]):
        result['label_id'][i] = int(1)
    elif (result['HATE'][i] >= result['OFFENSIVE'][i] and result['HATE'][i] >= result['CLEAN'][i]):
        result['label_id'][i] = int(2)


result.to_csv("Result.csv")





