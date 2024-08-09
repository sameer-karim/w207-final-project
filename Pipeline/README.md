# Step 0 Before run the pipeline
![Step 0](https://github.com/sameer-karim/w207-final-project/blob/7466ece89192c3ff84b88cabd007eac2929e4f7d/Pipeline/Step%200.JPG)

# 1. Data Preprocessing: 
This code will clean the raw data and generate the preprocessed data ready to load to the model for train

## python data_preprocessing.py

By running this code you will get a preprocessed_full_data_ready_for_model.csv. Besides you will also generate the raw test_data_X and test_data_Y which is used in Step3: model prediction and evaluation
![Step 1](https://github.com/sameer-karim/w207-final-project/blob/7466ece89192c3ff84b88cabd007eac2929e4f7d/Pipeline/Step%201.JPG)

# 2. Train Model:

## python model_training.py preprocessed_full_data_ready_for_model.csv

By running this code, it will generate the model.
![Step 2](https://github.com/sameer-karim/w207-final-project/blob/7466ece89192c3ff84b88cabd007eac2929e4f7d/Pipeline/Step%202.JPG)


# 3. Test/Predict using model
## python predict.py test_data_X.csv pretrained_model.joblib predictions.csv test_data_Y.csv
You will use model trained in step 2 and use input as test_data_X.csv which is the raw test data with [model	year	price	transmission	mileage	fuelType	tax	mpg	engineSize] columns to predict the car price.
The prediction from the model will be saved in predictions.csv. And this prediction result will be compared with the test_data_Y price result to evaluate the prediction performance.
![Step 3](https://github.com/sameer-karim/w207-final-project/blob/7466ece89192c3ff84b88cabd007eac2929e4f7d/Pipeline/Step%203.JPG)
