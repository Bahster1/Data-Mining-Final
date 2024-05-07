import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder


ss = StandardScaler()
le = LabelEncoder()

data = pd.read_csv('../../data/features_3_sec.csv')
data = data.drop(columns=['filename', 'length'])

X = data.drop(columns=['label'])
y = data['label']

le.fit(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = pd.get_dummies(y_train)
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

model = keras.models.Sequential()

model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(X_train_ss, y_train, validation_split=0.2, epochs=10, batch_size=32)

y_pred = model.predict(X_test_ss)
y_pred = np.argmax(y_pred, axis=1)
y_pred = le.inverse_transform(y_pred)

print(model.summary())
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()
