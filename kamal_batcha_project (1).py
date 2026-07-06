
import pandas as pd
from autogluon.tabular import TabularPredictor

# Create the train dataset in pandas by reading the csv
# Set the parsing of the datetime column so you can use some of the `dt` features in pandas later
train = pd.read_csv("train.csv")
train.head()

# Simple output of the train dataset to view some of the min/max/varition of the dataset features.
train.describe()

# Create the test pandas dataframe in pandas by reading the csv, remember to parse the datetime!
test = pd.read_csv("test.csv")
test.head()

# Same thing as train and test dataset
submission = pd.read_csv("sampleSubmission.csv")
submission.head()

"""## Step 3: Train a model using AutoGluon’s Tabular Prediction

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.
"""

drop_train_features=["casual","registered"]
train.drop(drop_train_features,axis=1,inplace=True)

train.info()

predictor=TabularPredictor(label="count",problem_type="regression",eval_metric="rmse").fit(train_data=train,time_limit=600,presets="best_quality")

"""### Review AutoGluon's training run with ranking of models that did the best."""

predictor.fit_summary()

"""### Create predictions from test dataset"""

predictor.leaderboard(silent=True).plot(kind="bar",x="model",y="score_val")

"""#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0."""

predictions = predictor.predict(test)
predictions = { 'count': predictions}
predictions = pd.DataFrame(data=predictions)
predictions.head()

predictions[predictions['count']<0]=0

submission["count"] = predictions
submission.to_csv("submission1.csv", index=False)



## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.
"""

train1=pd.read_csv("train.csv")
test1= pd.read_csv("test.csv")

"""## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.
"""

train1.loc[:, 'datetime'] = pd.to_datetime(train1.loc[:, 'datetime'])
test1.loc[:, 'datetime'] = pd.to_datetime(test1.loc[:, 'datetime'])

train1['year'] =  pd.to_datetime(train1.loc[:, 'datetime']).dt.year
train1['month'] =  pd.to_datetime(train1.loc[:, 'datetime']).dt.month
train1['day'] =  pd.to_datetime(train1.loc[:, 'datetime']).dt.day
train1['hour']=  pd.to_datetime(train1.loc[:, 'datetime']).dt.hour
test1['year'] =  pd.to_datetime(test1.loc[:, 'datetime']).dt.year
test1['month'] =  pd.to_datetime(test1.loc[:, 'datetime']).dt.month
test1['day'] =  pd.to_datetime(test1.loc[:, 'datetime']).dt.day
test1['hour']=  pd.to_datetime(test1.loc[:, 'datetime']).dt.hour

train1["season"]=train1["season"].astype("category")
train1["weather"]=train1["weather"].astype("category")
test1["season"]=test1["season"].astype("category")
test1["weather"]=test1["weather"].astype("category")

drop_train_features=["casual","registered"]
train1.drop(drop_train_features,axis=1,inplace=True)

# View are new feature
train1.head()

train1.hist()

"""## Step 5: Rerun the model with the same settings as before, just with more features"""

predictor_new_features =TabularPredictor(label="count",problem_type="regression",eval_metric="rmse").fit(train_data=train1,time_limit=700,presets="best_quality")

predictor_new_features.fit_summary()

predictions= predictor_new_features.predict(test1)
predictions = { 'count': predictions}
predictions = pd.DataFrame(data=predictions)
predictions.head()

# Remember to set all negative values to zero
predictions[predictions['count']<0]=0

submission_new_features=pd.read_csv("sampleSubmission.csv")

# Same submitting predictions
submission_new_features["count"] = predictions
submission_new_features.to_csv("submission_new_features.csv", index=False)


"""#### New Score of `?`

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.
"""

train2=pd.read_csv("train.csv")
test2= pd.read_csv("test.csv")

train2.loc[:, 'datetime'] = pd.to_datetime(train2.loc[:, 'datetime'])
test2.loc[:, 'datetime'] = pd.to_datetime(test2.loc[:, 'datetime'])

train2['year'] =  pd.to_datetime(train2.loc[:, 'datetime']).dt.year
train2['month'] =  pd.to_datetime(train2.loc[:, 'datetime']).dt.month
train2['day'] =  pd.to_datetime(train2.loc[:, 'datetime']).dt.day
train2['hour']=  pd.to_datetime(train2.loc[:, 'datetime']).dt.hour
test2['year'] =  pd.to_datetime(test2.loc[:, 'datetime']).dt.year
test2['month'] =  pd.to_datetime(test2.loc[:, 'datetime']).dt.month
test2['day'] =  pd.to_datetime(test2.loc[:, 'datetime']).dt.day
test2['hour']=  pd.to_datetime(test2.loc[:, 'datetime']).dt.hour

train2["season"]=train2["season"].astype("category")
train2["weather"]=train2["weather"].astype("category")
test2["season"]=test2["season"].astype("category")
test2["weather"]=test2["weather"].astype("category")

drop_train_features=["casual","registered","datetime"]
train2.drop(drop_train_features,axis=1,inplace=True)

drop_test_features=["datetime"]
test2.drop(drop_test_features,axis=1,inplace=True)

import autogluon.core as ag
hp_tune=True
rf_option={
   
   'n_estimators': [100,125,150],
   'max_depth': [5, 7, 10],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4],
   'max_features': ['auto', 'sqrt', 'log2']
}
gbm_options = {
    'num_boost_round':85,  
    'max_depth':5,
    
}
hyperparameters={
                 "GBM":gbm_options,"RF":rf_option,
}
num_trails=2
search_strategy = 'auto'

hyperparameter_tune_kwargs={
    "num_trails":num_trails,
    "searcher":search_strategy,
    "scheduler":"local",
}
predictor_new_hpo = TabularPredictor(label="count",problem_type="regression",eval_metric="rmse").fit(train_data=train2,time_limit=700,presets="best_quality",hyperparameters=hyperparameters,hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

predictor_new_hpo.fit_summary()

prediction_new_hpo = predictor_new_hpo.predict(test1)
prediction_new_hpo = { 'count': prediction_new_hpo}
prediction_new_hpo = pd.DataFrame(data=prediction_new_hpo)
prediction_new_hpo.head()

# Remember to set all negative values to zero

prediction_new_hpo[prediction_new_hpo['count']<0]=0

submission_new_hpo=pd.read_csv("sampleSubmission.csv")

# Same submitting predictions
submission_new_hpo["count"] = prediction_new_hpo
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
submission_new_hpo

"""#### New Score of `?`

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report
"""

# Taking the top model score from each training run and creating a line plot to show improvement
# You can create these in the notebook and save them to PNG or use some other tool (e.g. google sheets, excel)
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score": [1.79818, 0.64953 ,0.46423]
    }
).plot(x="model", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_train_score.png')

# Take the 3 kaggle scores and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.79818,0.64953,0.46423]
    }
).plot(x="test_eval", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_test_score.png')

"""### Hyperparameter table"""

# The 3 hyperparameters we tuned with the kaggle score as the result
hp=pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": ['default_vals', 'default_vals', 'n_estimators: 100,125,150'],
    "hpo2": ['default_vals', 'default_vals', 'num_boost_round:85'],
    "hpo3": ['default_vals', 'default_vals', 'max_depth:5, 7, 10']

})

hp.head()

