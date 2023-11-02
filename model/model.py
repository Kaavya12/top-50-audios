import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Reshape, Dropout

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

#gathering the data, check the mdeff/fma repo for this
features = pd.read_csv("./fma_metadata/features.csv", header=[0,1,2], index_col=0)
tracks = pd.read_csv("./fma_metadata/tracks.csv", header=[0,1], index_col=0)

tracks = tracks[tracks['track', 'genre_top'].notna()]
tracks = tracks[(tracks['track', 'genre_top'] != "Easy Listening") & (tracks['track', 'genre_top'] != "Soul-RnB")]

#checking the numbers of data points for each class
print(tracks.loc[:, [('track', 'genre_top')]].value_counts())

columns = ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr', 'tonnetz']
features_indexed = features.loc[tracks.index]

#preprocessing the data 
def preprocess(tracks, features, columns):
  enc = LabelEncoder()
  X = features[columns].sort_index()
  y = tracks['track', 'genre_top'].sort_index()
  X, y = shuffle(X, y, random_state=12)

  X_train, X_test,y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

  y_train = enc.fit_transform(y_train)
  y_test = enc.transform(y_test)

  pipe = Pipeline([
      ('scaler', StandardScaler(copy=False)),
      ('feature_selection', VarianceThreshold(threshold=0.5)),
  ])

  X_train = pipe.fit_transform(X_train, y_train)
  X_test = pipe.transform(X_test)

  return y_train, y_test, X_train, X_test, enc, pipe

y_train, y_test, X_train, X_test, enc, pipe = preprocess(tracks, features_indexed, columns)

joblib.dump(pipe, 'pipe.joblib')
joblib.dump(enc, 'enc.joblib')

#to print the shapes of the training and testing data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#checking the encoder's transformation and which numbers map to which labels
print(enc.inverse_transform(range(14)))

#creating weights to manage the imbalance in the dataset
classes, counts = np.unique(y_train, return_counts=True)
max_count = counts.max()
weights = {}

for i in range(len(counts)):
  count = counts[i]
  weights[i] = math.ceil(max_count/(count)+1) #the +1 creates a buffer in the weights
  
#creating the model
model = tf.keras.Sequential()
model.add(Input(shape=(X_train.shape[1])))
model.add(Reshape((X_train.shape[1], 1)))
model.add(Conv1D(128, 12, activation='relu')) #mod_4 had 128 mod_5 had 64
model.add(MaxPooling1D())
model.add(Conv1D(64, 12, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

#compiling the model with optimizers and loss
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=adam)

#setting callbacks for better performance
filepath='./model_checkpoint'
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)

#training the model
history = model.fit(X_train, y_train,
                    class_weight=weights,
                    validation_split = 0.1,
                    batch_size=2048,
                    epochs=100, verbose=1, 
                    callbacks=[es_callback, checkpoint])

#plotting model performance
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

#creating predictions on test data
preds_test = model.predict(X_test)
preds_test = np.argmax(preds_test, axis=1)

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
preds_onehot_test = label_binarizer.transform(preds_test)

print(roc_auc_score(y_onehot_test, preds_onehot_test, average=None))
print(roc_auc_score(y_onehot_test, preds_onehot_test, average='weighted'))

#confusion matrix to properly visualise the model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax)
plt.show()

#saving the model
model.save('model.h5')