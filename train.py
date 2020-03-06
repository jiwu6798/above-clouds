# -*- coding: utf-8 -*-
"""Copy of Copy of Model-classweight-qwk.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_MpUpKXyaKw4Q_sxPZIHuIdH6Zh442Q9
"""

# Commented out IPython magic to ensure Python compatibility.

import random
random.seed(42)
import pandas as pd 
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
from sklearn.utils import class_weight
import datetime
from tensorflow.keras.callbacks import TensorBoard
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tensorflow.keras.layers import Input


def preprocess_image(img, desired_size=299, tol=7):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #crop image from gray
    if img.ndim ==2:
      mask = img>tol
      return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
      gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      mask = gray_img>tol

    check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    if (check_shape == 0): # image is too dark so that we crop out everything,
      return img # return original image
    else:
      img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
      img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
      img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
      img = np.stack([img1,img2,img3],axis=-1)

    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), 30) ,-4 ,128)
    return img

training_data = pd.read_csv('../input/train.csv')
df = pd.DataFrame(training_data, columns=['id_code', 'diagnosis'])
df['diagnosis'] = df['diagnosis'].astype(str)
df['id_code'] = df['id_code'].astype(str) + '.png'

tr_df, test_df = train_test_split(df,test_size=0.1)
train_df, val_df = train_test_split(df,test_size=0.2)

train_datagen = ImageDataGenerator(rescale=1./255.,
                                  preprocessing_function = preprocess_image,
                                   brightness_range = [0.8,1.2],
                                   rotation_range=30,
                                   shear_range=0.15)
test_datagen=ImageDataGenerator(rescale=1./255.,
                                   preprocessing_function = preprocess_image)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory = "../input/train_images",
    x_col='id_code', 
    y_col='diagnosis',
    batch_size = 16,
    seed=42,
    target_size=(299, 299),
    shuffle=True)

valid_generator= train_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory = "../input/train_images",
    x_col = "id_code",
    y_col = "diagnosis",
    batch_size = 16,
    seed=42,
    target_size=(299, 299),
    shuffle=False)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="../input/train_images",
    x_col="id_code",
    y_col=None,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(299, 299))

class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=32, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  workers=1, use_multiprocessing=True,
                                                  verbose=1)
            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)
                # return np.sum(y.astype(int), axis=1) - 1
            
            score = cohen_kappa_score(self.y_val,
                                      flatten(y_pred),
                                      labels=[0,1,2,3,4],
                                      weights='quadratic')

            print("\n epoch: %d - Kappa_score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('save checkpoint: ', score)
                self.model.save('./models/model_newdata.h5')

qwk = QWKEvaluation(validation_data=(valid_generator, val_df['diagnosis'].astype(int)),
                    batch_size=32, interval=1)



class_weights = [0.27159274,2.92515905, 1.32052219,8.33212521, 10.1152]
#class_weights = {0:0.27159274,1:2.92515905, 2:1.32052219,3:8.33212521, 4:10.1152}


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'fit/ ' + current_time + 'model_newdata'
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch')

#from tensorflow.keras.optimizers import Adam

def build_model():  
  base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                     weights='imagenet',
                                                     input_tensor=Input(shape=(299,299, 3)),
                                                     input_shape=None,
                                                     pooling=None,
                                                     classes=5)
  x = base_model.output
  x = tf.keras.layers.BatchNormalization(name='bn1')(x)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dense(1024, activation='relu', name='fc1024')(x)
  x = tf.keras.layers.BatchNormalization(name='bn2')(x)
  x = tf.keras.layers.Dense(512, activation='relu', name='fc512')(x)
  x = tf.keras.layers.BatchNormalization(name='bn3')(x)
  x = tf.keras.layers.Dense(256, activation='relu', name='fc256')(x)
  x = tf.keras.layers.BatchNormalization(name='bn4')(x)
  x = tf.keras.layers.Dense(128, activation='relu', name='fc128')(x)
  x = tf.keras.layers.BatchNormalization(name='bn5')(x)
  x = tf.keras.layers.Dense(64, activation='relu', name='fc64')(x)
  x = tf.keras.layers.BatchNormalization(name='bn6')(x)
  x = tf.keras.layers.Dense(5, activation='softmax', name='predictions')(x)
  model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

  for layer in model.layers:
      layer.trainable = True
  
  model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

## build
model = build_model()

history = model.fit_generator(
    train_generator,
    epochs=3,
    verbose=1,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    class_weight='auto',
    callbacks=[qwk,tensorboard_callback])



"""## accuracy"""

test_generator.reset()
model = load_model('model_newdata.h5')
pred = model.predict_generator(test_generator,verbose=1,steps=test_generator.n)
predicted_class = np.argmax(pred, axis = 1)

true_class=np.asarray(list(map(int, test_df['diagnosis'].tolist())))
print((predicted_class == true_class).sum()/len(predicted_class))

from sklearn.metrics import classification_report
print(classification_report(true_class, predicted_class))

print(cohen_kappa_score(
            true_class,
            predicted_class, 
            weights='quadratic'
        ))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(true_class, predicted_class))

