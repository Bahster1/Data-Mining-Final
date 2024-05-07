import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras

df = pd.read_csv('../../data/spotify_songs.csv')
df = df.drop(columns=['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name',
                      'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_subgenre',
                      'track_popularity'])

X = df.drop(columns=['playlist_genre'])
y = pd.get_dummies(df['playlist_genre'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.models.Sequential()

model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(6, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))
