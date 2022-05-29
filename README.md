# Data-Challenge
## COVID19 Retweet Prediction

### Instructions
In order to run the code there has to be a folder named 'data' in the same folder with the python scripts that containes the original csv files (train.csv and evaluation.csv).

To execute the preprocessing on the files you can run preprocessing_main.py. 
The preprocessing of the data takes approximately an hour to execute so we have created two new csv files that contain the preprocessed data. 
You can find them here https://drive.google.com/drive/folders/1ezxXfNLQlS5SScwTwnyajDb3bmz5ui2C?usp=sharing . 

The preprocessed data have to be in the same folder with the python scripts (not in the data folder). 
If you choose to get the preprocessed data by running the code instead they will appear in the correct place.

For the PCA you can execute the code PCA.py. 
Our mlp model can be found on mlp.py. 
Our Random Forest model can be found on random_forest.py. 
After the execution a file named eval_predictions.txt, that contains the predictions for the evaluation data, will appear in the folder. 
Please note that due to batches the results are not always the same.
