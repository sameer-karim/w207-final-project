# Predicting Used Car Prices
## Sameer Karim, Zeija Jing, and Ricky Pang

Final code file: Combined_All_LR_RF_FNN_XGBoost_Model_Code.ipynb (All of our work and progress was tracked on the project branch, and then pushed to main once it all was finalized.)

### Table of Contents:
- Introduction & Relevance
- Setup and Installation
- Data
- Model Training and Evaluation
- Conclusion

### Introduction & Relevance:
In this machine learning project, we aim to predict the prices of used cars using various features found in a Kaggle dataset (link below). As far as financial planning goes - both sellers and buyers benefit from knowing the value of a used car. Platforms like Kelly Blue Book have been made popular for this very reason. Sellers can look to capture maximum value for their vehicle and buyers can look to acquire a new car at a reasonable price. Depreciation trends and understanding how cars either lose or gain value over time is also valuable to both buyers and sellers. A car’s value tends to exponentially decay, especially for luxury cars and buyers may look to purchase these types of cars immediately after the biggest drop in value has occurred. On the other hand some cars actually increase in value over time and sellers may look to hold on to them for as long as possible. Insurance companies also need to know the value of cars so that they can set premiums and so that they do not take on too much risk in the form of claim payouts. Banks and financial institutions that issue loans need to be able to predict a car’s value so that they can determine loan amounts and prevent people from defaulting on loans. All of these insights are reasons why accurately predicting a car’s value is important. 

### Setup & Installation:
Clone the repo and run the predict.py script with a designated file path for the data you would like to use to predict.

### Data:

#### Data Introduction:
- Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data?select=vw.csv

This data was derived from scraping used car listings. After concatening all of the csv files into one dataframe, added a brand column that keeps track of the manufacturer of the vehicle as well as EDA to visualize any outliers or trends. We found that there were over 9000 rows of missing data for the 'mpg' and 'tax' categories - multiple imputation strategies were implemented to remedy this. We tried dropping columns, imputing with the median, and imputing with a KMeans cluster average. 

#### Feature Engineering
Our categorical data encoding strategies included label, frequency, ordinal, and one-hot encoding. We settled on one-hot encoding because it made the most sense for the types of categorical features in the dataset. In regards to numerical scaling – we first tried log transforming features that were right-skewed or not normal. We also tried Z-Scaling using StandardScaler. We found that performance was negatively impacted by scaling so we ended up leaving the numerical features unscaled. 

### Model Training and Evaluation:

#### Metrics
The primary metric used to quantify our model's predictions will be the Mean Absolute Percentage Error (MAPE). MAPE is often chosen for regression problems for a few reasons: It is easily accessible and understood by a wide variety of stakeholders; a MAPE of 7% indicates that, on average, the models predictions are off by 7%. It also is effective in dealing with features that span multiple orders of magnitude - ensuring errors are scaled appropriately. 

#### Model Selection & Implementation:
Because each team member wanted practice in building machine learning models, our approach had each team member build their own machine learning model using MAPE as our evaluation metric. We chose to explore a Linear Regression, Random Forest Regressor, and a feedforward Neural Network. 

In our Linear Regression model, we used np.log(price) to ensure that predictions are positive. The results are quite straightforward, we fit the model on our cleaned test data and returned the following results:

- MAPE: 9.64%
- R<sup>2</sup>: 92.92%

Our Random Forest model utilized a parameter grid and RandomizedSearchCV to explore different combinations of hyperparameters. RandomizedSearchCV was used over GridSearchCV to emphasize efficiency and iterability. The best estimator from the search was selected and trained on the training set. The model performed well, achieving a 7.39% MAPE value on the test set. We iterated on the model by creating a more focused parameter grid around the best parameters obtained from the first run, and running GridSearchCV, a more exhaustive search. Results yielded a 7.28% MAPE value - a marginal improvement. Further investigation would be needed to find where the reason lies; this may be due to the model complexity or the bias-variance tradeoff. Another further step could involve creating new features, such as interaction terms, polynomial features, or domain-specific features. 

The XGBoost model was our highest achiever reaching a 6.5% MAPE value on the test data.
The model utilized the 'gbtree' booster, with the 'reg' objective to minimize squared error for regression. We set an initial learning rate (eta) of 0.1, a maximum tree depth (max_depth) of 20, and both subsample and colsample_bytree values set to 0.8 to ensure robust and diverse trees.

### Performance Evaluation
Mean Absolute Percentage Error of Various Models with and without K-fold Cross-Validation (k=5) for Training and Testing Phases. 

![Performance Evaluation](https://github.com/sameer-karim/w207-final-project/blob/ad31a071bae0bef233cb3c16d72ea85c9cc5be56/Performance%20Evaluation.JPG)

From the performance summary, it can be seen that XGBoost does have the lowest values of MAPE, so essentially, it is the best among the models tested with a slight increase in MAPE when going from training to testing, which might indicate some extent of overfitting.

Random Forest also goes well, with a bit higher MAPE values compared to XGBoost but still much better than Linear Regression and the Base model. Linear Regression has higher MAPE values, thus it might not be that good as these two: XGBoost and Random Forest in this particular task. However, it is still pretty impressive compared to the Baseline model. All the three models beat the baseline performance significantly.

K-fold cross-validation stabilizes the performance of the model and provide better generalization under certain circumstances. The gapdifference between train and test MAPE—is the smallest for Linear Regression, so it generalizes relatively well but with higher error. For XGBoost and Random Forest, this gap is more significant, much more without cross-validation, thus putting forward a tradeoff between the model complexity and the risk of overfitting.
### Conclusion
Despite our diverse approaches in modeling our data, we arrived at similar metrics. Each model performs exceptionally well on unseen data, particularly the XGBoost model (6.5% MAPE without Kfold and 6.29% with Kfold). In hindsight, this aligns with our other methods of modeling, as XGBoost combines aspects of efficiency, regularization, and flexibility to create an effective implemention of gradient boosting. Further research on ensemble models, feature engineering, and other hyperparameter tuning methods could improve the performance of this model, however given our time and resource constraints, we are satisfied with the results and hope you learn from our findings! 
