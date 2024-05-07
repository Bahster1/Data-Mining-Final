import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../../data/spotify_songs.csv')
df = df.drop(columns=['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name',
                      'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_subgenre',
                      'track_popularity'])

X = df.drop(columns=['playlist_genre'])
y = df['playlist_genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
	'weights': ['uniform', 'distance'],
	'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid = GridSearchCV(estimator=knn, param_grid=params, verbose=1)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
