# w207-final-project
Sameer Karim, Zeija Jing, and Ricky Pang's w207 final project 

Car Price Prediction

Table of Contents:
Introduction
Setup and Installation
Data Preparation
Model Training and Evaluation
Hyperparameter Tuning
Conclusion

Introduction:
This project aims to predict the prices of used cars using various machine learning models. The primary goal is to minimize the Mean Absolute Percentage Error (MAPE) of the predictions. Models explored include Linear Regression, Random Forest, Feedforward Neural Networks, LightGBM, CatBoost, and XGBoost.

For our project we have chosen to explore a labeled dataset that stores features of a car and a label that records the price of that car. There are a number of reasons why we should be interested in predicting the value of a used car based on its features. As far as financial planning goes both sellers and buyers benefit from knowing the value of a used car. Platforms like kelly blue book have been made popular for this very reason. Sellers can look to capture maximum value for their vehicle and buyers can look to acquire a new car at a reasonable price. Insurance companies also need to know the value of cars so that they can set premiums and so that they do not take on too much risk in the form of claim payouts. Depreciation trends and understanding how cars either lose or gain value over time is also valuable to both buyers and sellers. A car’s value tends to exponentially decay, especially for luxury cars and buyers may look to purchase these types of cars immediately after the biggest drop in value has occurred. On the other hand some cars actually increase in value over time and sellers may look to hold on to them for as long as possible. Lastly, financial institutions that issue loans need to be able to predict a car’s value so that they can determine loan amounts and prevent people from defaulting on loans. All of these insights are reasons why accurately predicting a car’s value is important. 

Below are some general motivations for this project:

Economic Impact: The used car market is a significant part of the automotive industry, impacting manufacturers, dealers, and consumers. Accurate valuation helps in making informed buying and selling decisions.

Financial Planning: Consumers often rely on the resale value of their current vehicles when planning to purchase a new car. Knowing the expected depreciation helps in budgeting and financial planning.

Insurance: Insurance companies need accurate car valuations to set premiums and determine payouts in case of claims. Incorrect valuations can lead to financial losses or higher costs for policyholders.

Loan Approvals: Financial institutions use car valuations to approve loans and determine loan amounts. Accurate predictions reduce the risk of defaults by ensuring the loan amount aligns with the car's value.

Depreciation Management: Car manufacturers and leasing companies need to understand depreciation trends to set appropriate lease rates and residual values, ensuring profitability and competitive pricing.

Market Stability: Accurate valuations contribute to market stability by preventing price bubbles and crashes. They ensure that car prices reflect true market conditions and supply-demand dynamics.

Consumer Confidence: Transparent and reliable car valuations increase consumer confidence in the used car market. Buyers and sellers are more likely to engage in transactions when they trust the valuation process.

Regulatory Compliance: In some regions, accurate car valuations are required for tax calculations and regulatory compliance. Ensuring accuracy helps in adhering to legal standards and avoiding penalties.


Setup:
Clone the repo and run the predict.py script with a designated file path for the data you would like to use to predict.

Data Preparation and Feature Engineering:
Data Introduction:
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data?select=vw.csv
We initially tried to use a popular used car dataset from Kaggle but found that the price values were very inaccurate - our metrics for performance across all models for this dataset were large, inaccurate, and unreasonable 
We found a new dataset that upon inspection seems to be a much more reasonable set of pricing listings for cars on the market 
The data is split up by brand so we needed to combine it all together
The columns were Brand, price, engine size, tax, mileage, transmission, fuel type, mpg, model, year
Overall this was a very friendly dataset to work with. We added a brand column that keeps track of the manufacturer of the vehicle. We also did some EDA to visualize anomalies in the data. We found that there were a number of 0s and missing data for a couple categories so we tried multiple imputation strategies to remedy that. We tried dropping the columns, imputing with the median, and imputing with a KMeans cluster average. For the KMeans cluster averaging we used the elbow method which tests different numbers of centroids and we plotted the WCSS. WCSS is within cluster sum of squares - which summarizes the total variance in a cluster. We then fit the data and assigned cluster labels to each row in the data. For missing values in tax and engine size we computed the average value in the cluster not including zeroes and replaced the missing values with the cluster average

 In addition we tried multiple strategies for encoding categorical data including label, frequency, ordinal, and one-hot encoding. We settled on one-hot encoding because it made the most sense for the types of categorical features in the dataset. Lastly we scaled the numerical features. We tried multiple approaches for scaling – we first tried log transforming features that were right-skewed or not normal. We also tried Z-Scaling using StandardScaler. We found that performance was negatively impacted by scaling so we ended up leaving the numerical features unscaled.

Model Training and Evaluation:

Metrics: 
Why Use MAPE for Used Car Price Prediction:
Interpretability: MAPE (Mean Absolute Percentage Error) provides an intuitive percentage error, making it easy to understand how far off predictions are on average relative to actual values.
Relative Error Measurement: MAPE expresses errors as a percentage, which is useful for comparing performance across different datasets or models.

Other Metrics to Consider:
MAE (Mean Absolute Error): Measures the average absolute errors between predicted and actual values, providing a straightforward error magnitude.

RMSE (Root Mean Squared Error): Penalizes larger errors more than MAE, useful for emphasizing the impact of significant prediction errors.

R² (R-squared): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables, showing model fit quality.

Adjusted R²: Adjusts R² for the number of predictors in the model, useful for comparing models with different numbers of predictors.

Model Selection:
We started off with a baseline of a simple regression we achieved right around 28% MAPE using only the numerical features
We set a goal of achieving at least 7% MAPE to match or beat the top rated code on Kaggle 
Our second model choice was a linear regression model including all the numerical features and the categorical features in their one hot representations we achieved 18% MAPE - this demonstrates the importance of including categorical data
Our next model of choice was a random forest regressor the MAPE was 7% this is already very ideal and matches other data scientists on Kaggle, we chose a random forest for its ability to capture non-linear relationships and interactions between features.
We tried feed forward neural networks of several configurations including varying sizes and amounts of hidden layers, dropout layers, and early stopping - these performed in the range of (10-20% MAPE). We chose FNNs due to their flexibility and ability to model complex, non-linear relationships. FNNs can learn intricate patterns in the data through multiple layers and neurons, potentially capturing dependencies that simpler models might miss. Additionally, with techniques like dropout and early stopping, FNNs can generalize well and avoid overfitting, making them suitable for varied datasets with complex interactions.
We also tried a few other production grade machine learning models including LightGBM and CatBoost. We chose to try these because they can handle categorical features without extensive preprocessing. They also utilize gradient boosting which allow them to achieve better performance with lower computational cost and faster training times. These models performed in the range of 10-20% MAPE 
The best model we used was XGBoost - it achieved a 6.5% MAPE on the test data
The default hyperparameters for XGBoost actually ended up working the best - we tried running both a random search and grid search around parameters that were in close proximity to the default and we did not have success in lowering MAPE


XGBoost Tuning:
Learning Rate (eta):
Description: Controls the step size at each iteration while moving towards a minimum of the loss function.
Why Vary: A lower learning rate ensures that the model learns more slowly and steadily, which can improve accuracy but requires more boosting rounds. A higher learning rate speeds up training but might overshoot the optimal solution.

Number of Boosting Rounds (n_estimators):
Description: The number of trees to be built.
Why Vary: More boosting rounds can lead to better performance but also increase the risk of overfitting. Fewer rounds may result in underfitting.

Max Depth (max_depth):
Description: The maximum depth of a tree.
Why Vary: Deeper trees can model more complex patterns but are more prone to overfitting. Shallower trees may not capture all the interactions in the data.

Min Child Weight (min_child_weight):
Description: Minimum sum of instance weight (hessian) needed in a child.
Why Vary: Higher values prevent the model from learning overly specific patterns (overfitting), while lower values allow for learning more complex relationships.

Subsample:
Description: The fraction of samples used for fitting individual base learners.
Why Vary: Reducing this fraction can prevent overfitting by introducing more noise into the training process but may also reduce the model's ability to capture the underlying data patterns.

Colsample_bytree, Colsample_bylevel, Colsample_bynode:
Description: Fraction of features to be randomly sampled for each tree (bytree), level (bylevel), or node (bynode).
Why Vary: Adjusting these parameters helps in managing overfitting and can make the model more robust by forcing it to consider different subsets of features.

Gamma (min_split_loss):
Description: Minimum loss reduction required to make a further partition on a leaf node.
Why Vary: A higher gamma value makes the algorithm more conservative and can help in reducing overfitting.

Lambda (reg_lambda) and Alpha (reg_alpha):
Description: L2 and L1 regularization terms, respectively.
Why Vary: Regularization parameters help in controlling the complexity of the model. Varying them helps in reducing overfitting by penalizing large coefficients.

By tuning these hyperparameters, one can control the complexity and behavior of the XGBoost model, balancing the trade-off between bias and variance, and thus improving predictive performance. The goal of hyperparameter tuning is to find the best set of parameters that minimize the error on unseen data.

Conclusion:

We chose a good dataset 
We cleaned and processed the data 
We encoded and scaled the data
We experimented with a bunch of models
Once we found a model of choice we tuned the model until we exceeded Kaggle standards
We built a predict.py script using the encoder, scaler, and model artifact





