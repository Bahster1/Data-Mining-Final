import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../data/features_3_sec.csv')
df = df.drop(columns=['length', 'filename'])

X = df.drop(columns=['label'])
y = df['label']
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

params = {
	'criterion': ['gini', 'entropy', 'log_loss']
}

dt = DecisionTreeClassifier()
grid = GridSearchCV(estimator=dt, param_grid=params, verbose=1)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()