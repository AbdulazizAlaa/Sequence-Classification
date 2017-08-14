
# Problem of Sequance Classification
 Classifying whether a review is positive or negative


```python
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(7)
```


```python
# load imdb dataset and only load top n words and zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```


```python
# pad and truncate input sequances
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```


```python
# model define
embedding_vector_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64, verbose=1)
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_6 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500, 32)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 500, 32)           3104      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0         
    _________________________________________________________________
    lstm_4 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 216,405
    Trainable params: 216,405
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/10
    25000/25000 [==============================] - 496s - loss: 0.4948 - acc: 0.7415   
    Epoch 2/10
    25000/25000 [==============================] - 450s - loss: 0.3800 - acc: 0.8337   
    Epoch 3/10
    25000/25000 [==============================] - 437s - loss: 0.2847 - acc: 0.8851   
    Epoch 4/10
    25000/25000 [==============================] - 433s - loss: 0.2526 - acc: 0.9008   
    Epoch 5/10
    25000/25000 [==============================] - 422s - loss: 0.2282 - acc: 0.9106   
    Epoch 6/10
    25000/25000 [==============================] - 418s - loss: 0.2075 - acc: 0.9201   
    Epoch 7/10
    25000/25000 [==============================] - 418s - loss: 0.1925 - acc: 0.9262   
    Epoch 8/10
    25000/25000 [==============================] - 421s - loss: 0.1798 - acc: 0.9320   
    Epoch 9/10
    25000/25000 [==============================] - 421s - loss: 0.1602 - acc: 0.9404   
    Epoch 10/10
    25000/25000 [==============================] - 423s - loss: 0.1528 - acc: 0.9430   





    <keras.callbacks.History at 0x7f97bb3a2ef0>




```python
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

```

    25000/25000 [==============================] - 183s   
    Accuracy: 87.74%



```python

```
