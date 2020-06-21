from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(data=X, columns=load_boston().feature_names)
y = pd.DataFrame(data=y, columns=["Price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
rf = RandomForestRegressor()
best_rf = RandomForestRegressor(max_depth=7, n_estimators=70, max_features=7, min_samples_leaf=1)
best_rf.fit(X_train, y_train)
y_predicted = best_rf.predict(X_train)
filename = 'finalized_model.pkl'
pickle.dump(best_rf, open(filename, 'wb'))
