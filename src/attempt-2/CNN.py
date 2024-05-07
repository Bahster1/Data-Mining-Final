import numpy as np
import pandas as pd
import keras
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

X = []
y = []

for root, dirs, files in os.walk('../../data/images_original/'):
	for d in dirs:
		for file in os.listdir(os.path.join(root, d)):
			img = cv2.imread(os.path.join(root, d, file), cv2.IMREAD_COLOR)
			X.append(img)
			y.append(d)

le.fit(y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = pd.get_dummies(y_train)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=3, kernel_size=(9, 9), activation='relu'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=3, kernel_size=(6, 6), activation='relu'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = le.inverse_transform(y_pred)

print(model.summary())
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()
