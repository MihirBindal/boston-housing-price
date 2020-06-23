from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(data=X, columns=load_boston().feature_names)
y = pd.DataFrame(data=y, columns=["Price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
rf = LinearRegression()
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_train)
filename = 'finalized_model.pkl'
pickle.dump(rf, open(filename, 'wb'))
