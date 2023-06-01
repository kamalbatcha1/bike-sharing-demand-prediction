# Report: Predict Bike Sharing Demand with AutoGluon Solution

H.Kamal Batcha
## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Initial model We are predicting count, so set the label as count and I am not adding any features and before submitting the model all negative value set to 0 because Kaggle will reject the submission if we don't set everything to be > 0 and I got a score of 1.79818

### What was the top ranked model that performed?
After hyperparameter tuning the hpo model got 0.46423

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
In the first model I got 1.79818 and I separate out the datetime into hour, day, month, year after adding these feature and season,weather dtype as category and I trained again and I got 0.64953 
### How much better did your model preform after adding additional features and why do you think that is?
after adding features my model got tremendous score that is 1.79818 to 0.64953 
## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
after adding the features and next is hyperparameter optimization so first i used rf model having 'n_estimators': [100,125,150] ,'max_depth': [5, 7,10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['auto', 'sqrt', 'log2'] and gbm having  'num_boost_round':85, 'max_depth':5 and hyperparameter_tune_kwargs having num_trails:2 and after fine tuning I got a score of 0.46423 

### If you were given more time with this dataset, where do you think you would spend more time?
I spend more time in hyperparameter optimization I tune this model using several hyperparameter to find out which is better 
### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|'default_vals'|'default_vals'|'default_vals'|1.79818|
|add_features|'default_vals'|'default_vals'|'default_vals'|0.64953|
|hpo|'n_estimators: 100,125,150'|'max_depth:5, 7, 10'|'num_boost_round:85'|0.46423|

### Create a line plot showing the top model score for the three (or more) training runs during the project.


![kscore.png](https://github.com/kamalbatcha1/bike-sharing-demand-prediction/blob/main/kscore.png)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![add 3 features.png](https://github.com/kamalbatcha1/bike-sharing-demand-prediction/blob/main/add%203%20features.png)

## Summary
In the initial model I got a score of 1.79818 after seperate out datetime feature into year,month,day, hour and season & weather as dtype category after adding these feature and I got a score 0.64953 and after hyper parameter tuning the model is far better score of 0.46423
