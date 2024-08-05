# Predicting Used Car Prices
## Sameer Karim, Zeija Jing, and Ricky Pang

### Table of Contents:
- Introduction & Relevance
- Setup and Installation
- Data Preparation
- Model Training and Evaluation
- Hyperparameter Tuning
- Conclusion

### Introduction & Relavance:
In this machine learning project, we aim to predict the prices of used cars using various features found in a Kaggle dataset (link below). As far as financial planning goes - both sellers and buyers benefit from knowing the value of a used car. Platforms like Kelly Blue Book have been made popular for this very reason. Sellers can look to capture maximum value for their vehicle and buyers can look to acquire a new car at a reasonable price. Depreciation trends and understanding how cars either lose or gain value over time is also valuable to both buyers and sellers. A car’s value tends to exponentially decay, especially for luxury cars and buyers may look to purchase these types of cars immediately after the biggest drop in value has occurred. On the other hand some cars actually increase in value over time and sellers may look to hold on to them for as long as possible. Insurance companies also need to know the value of cars so that they can set premiums and so that they do not take on too much risk in the form of claim payouts. Banks and financial institutions that issue loans need to be able to predict a car’s value so that they can determine loan amounts and prevent people from defaulting on loans. All of these insights are reasons why accurately predicting a car’s value is important. 

### Setup:
Clone the repo and run the predict.py script with a designated file path for the data you would like to use to predict.

### Data Preparation and Feature Engineering:

#### Data Introduction:
- Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data?select=vw.csv

This data was derived from scraping used car listings. After concatening all of the csv files into one dataframe, added a brand column that keeps track of the manufacturer of the vehicle as well as EDA to visualize any outliers or trends. We found that there were over 9000 rows of missing data for the 'mpg' and 'tax' categories - multiple imputation strategies were implemented to remedy this. We tried dropping columns, imputing with the median, and imputing with a KMeans cluster average. 

Once our data was split into a training, validation, and test set, (60/20/20) we implemented our feature scaling and encoding strategies. Our categorical data encoding strategies included label, frequency, ordinal, and one-hot encoding. We settled on one-hot encoding because it made the most sense for the types of categorical features in the dataset. In regards to numerical scaling – we first tried log transforming features that were right-skewed or not normal. We also tried Z-Scaling using StandardScaler. We found that performance was negatively impacted by scaling so we ended up leaving the numerical features unscaled. 

### Model Training and Evaluation:

#### Metrics
The primary metric used to quantify our model's predictions will be the Mean Absolute Percentage Error (MAPE). MAPE is often chosen for regression problems for a few reasons: It is easily accessible and understood by a wide variety of stakeholders; a MAPE of 7% indicates that, on average, the models predictions are off by 7%. It also is effective in dealing with features that span multiple orders of magnitude - ensuring errors are scaled appropriately. 

#### Model Selection & Implementation:
Because each team member wanted practice in building machine learning models, our approach had each team member build their own machine learning model using MAPE as our evaluation metric. We chose to explore a Linear Regression, Random Forest Regressor, and a feedforward Neural Network. 

Our baseline modeling exploration on strictly numerical data outputted a 28% MAPE value. After one-hot encoding our categorical data and including in the linear regression, we achieved an 18% MAPE value.

Our Random Forest model utilized a parameter grid and RandomizedSearchCV to explore different combinations of hyperparameters. RandomizedSearchCV was used over GridSearchCV to emphasize efficiency and iterability. The best estimator from the search was selected and trained on the training set. The model performed well, achieving a 7.39% MAPE value on the test set.

We iterated on the model by creating a more focused parameter grid around the best parameters obtained from the first run, and running GridSearchCV, a more exhaustive search. Results yielded a 7.33% MAPE value - a marginal improvement. This may be due to the model complexity or the bias-variance tradeoff. Another further step could involve creating new features, such as interaction terms, polynomial features, or domain-specific features. 

The best model we used was XGBoost - it achieved a 6.5% MAPE on the test data
The default hyperparameters for XGBoost actually ended up working the best - we tried running both a random search and grid search around parameters that were in close proximity to the default and we did not have success in lowering MAPE


Conclusion:

We chose a good dataset 
We cleaned and processed the data 
We encoded and scaled the data
We experimented with a bunch of models
Once we found a model of choice we tuned the model until we exceeded Kaggle standards
We built a predict.py script using the encoder, scaler, and model artifact





