
#Diabetic Retinopathy Detection Convolutional Neural Network Model
This project is to build a convolutional neural network to detect Diabetic Retinopathy by the presence of lesions associated with the vascular abnormalities caused by the disease. And then make a prediction on lesions image, it will contribute to Diabetic retinopathy treatment。

#Algorithm
This folder contains the python programs train.py and prediction.py
#
train.py: To train model, download the data set from kaggle("https://www.kaggle.com/c/aptos2019-blindness-detection/") and unzip the train_images folder under input folder.
"Train.csv" should be in this directory as well. To run this file, use *python train.py". The default output is a h5 file named "model_newdata.h5"

prediction.py: To predict label of data. To run this file, use *python predict.py". The default output is a txt file named "Pred.txt"