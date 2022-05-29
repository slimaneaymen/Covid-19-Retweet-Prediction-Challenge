from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


#Read train data
train_data = pd.read_csv(r"/preprocessed_train70.csv", encoding ='latin1', keep_default_na=False, index_col=0)
test_data = pd.read_csv(r"/preprocessed_train30.csv", encoding ='latin1', keep_default_na=False, index_col=0)
y_test=pd.read_csv(r"/Y_test.csv", encoding ='latin1', keep_default_na=False)
y_train=pd.read_csv(r"/Y_train.csv", encoding ='latin1', keep_default_na=False)
X_train = train_data.drop(['hashtags_count'], axis=1)
X_train = X_train.drop(['user_verified'], axis=1)
X_train = X_train.drop(['mentions_count'], axis=1)
X_train = X_train.drop(['urls_count'], axis=1)
X_train = X_train.drop(['urls_freq'], axis=1)
X_train = X_train.drop(['mentions_freq'], axis=1)
X_train = X_train.drop(['v4'], axis=1)
X_train = X_train.drop(['v3'], axis=1)
X_train = X_train.drop(['v2'], axis=1)
X_train = X_train.drop(['v1'], axis=1)
y=np.zeros((y_train.shape[0],))
y=y_train.to_numpy()
y=y.reshape(y_train.shape[0],)

def random_forest(n_est,max_dep,train_data, y,test_data,test=False,y_test):
    # Fit the grid search to the data
    rf = RandomForestRegressor(n_estimators = n_est,max_depth=max_dep,random_state = 42,n_jobs=-1) # Train the model on training data
    rf.fit(train_data, y)
    # Use the forest's predict method on the test data
    y_pred = rf.predict(test_data)
    # Calculate the absolute errors
    importances = list(rf.feature_importances_)
    # Performance metrics
    if (test):
        print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    # Dump the results into a file that follows the required Kaggle template
    with open('eval_predictions_1.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for index, prediction in enumerate(y_pred):
            writer.writerow([str(test_data.index.values[index]) , str(int(prediction))])

random_forest(n_est,max_dep,train_data, y,test_data,test=False,y_test)
