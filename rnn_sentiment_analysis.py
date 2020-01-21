# import the built-in imdb dataset in Keras
from keras.datasets import imdb
import numpy as np  

# Set the vocabulary size
vocabulary_size = 5000

old = np.load
np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)

# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

np.load = old
del(old)

# Inspect a sample review and its label
# print("--- Review ---")
# print(X_train[5])
# print("--- Label ---")
# print(y_train[5])
# Map word IDs back to words
# word2id = imdb.get_word_index()
# id2word = {i: word for word, i in word2id.items()}
# print("--- Review (with words) ---")
# print([id2word.get(i, " ") for i in X_train[7]])
# print("--- Label ---")
# print(y_train[7])

from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 500

# Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Design your model
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# Compile your model, specifying a loss function, optimizer, and metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Specify training parameters: batch size and number of epochs
batch_size = 64
num_epochs = 3

# Reserve/specify some training data for validation (not to be used for training)
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]  # first batch_size samples
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]  # rest for training

# Train your model
model.fit(X_train2, y_train2,
          validation_data=(X_valid, y_valid),
          batch_size=batch_size, epochs=2)

# Save your model, so that you can quickly load it in future (and perhaps resume training)
model_file = "rnn_model.h5"  # HDF5 file
model.save(model_file)

# Evaluate your model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)  # returns loss and other metrics specified in model.compile()
print("Test accuracy:", scores[1])  # scores[1] should correspond to accuracy if you passed in metrics=['accuracy']





