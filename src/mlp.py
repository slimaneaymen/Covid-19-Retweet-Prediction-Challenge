import pandas as pd
import csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error

def mlp(X_train, Y_train, X_test, fileName, test=False, Y_test=None):

    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='linear'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=30, batch_size=100, verbose=False)

    y_pred = model.predict(X_test)
    y_pred = [0 if x < 0 else x for x in y_pred]
    
    if(test):
        print("Mean Absolute Error: ", mean_absolute_error(Y_test, y_pred))

    # Dump the results into a file that follows the required Kaggle template
    with open(fileName, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for index, prediction in enumerate(y_pred):
            writer.writerow([str(X_test.index.values[index]) , str(int(prediction))])

    return history

#Read train data
train_data = pd.read_csv("preprocessed_train.csv", keep_default_na=False, index_col=0)
test_data = pd.read_csv("preprocessed_eval.csv", keep_default_na=False, index_col=0)


train_data = train_data.drop(['user_followers_count'], axis=1)
test_data = test_data.drop(['user_followers_count'], axis=1)
train_data = train_data.drop(['user_friends_count'], axis=1)
test_data = test_data.drop(['user_friends_count'], axis=1)
train_data = train_data.drop(['user_statuses_count'], axis=1)
test_data = test_data.drop(['user_statuses_count'], axis=1)


data = pd.read_csv("data/train.csv", keep_default_na=False)
Y_train = data['retweet_count']

print("MultiLayer Perceptron")
history = mlp(train_data, Y_train, test_data, 'eval_predictions.txt')

plt.figure()
plt.title("Mean Absolute Error")
plt.plot(history.history['loss'])
plt.show()
