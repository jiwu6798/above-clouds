import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# preparing data
training_data = pd.read_csv('../input/train.csv')
total_label = []
for data in training_data:
    total_label.extend(data[1])
df = pd.DataFrame(training_data, columns=['id_code', 'diagnosis'])

#df.loc[df.level > 0, 'level'] = 1
df['diagnosis'] = df['diagnosis'].astype(str)
df['id_code'] = df['id_code'].astype(str) + '.png'

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    ##predict data from 30000th data point
    ##if needed, predict all by changing df[30000:] to df
    dataframe=df[30000:], 
    directory="../input/train_images",
    x_col="id_code",
    batch_size=1,
    class_mode=None,
    target_size=(299, 299))

"""## accuracy"""
model = tf.keras.models.load_model('model_newdata.h5')
pred_prob = model.predict_generator(test_generator, verbose=1, steps=test_generator.n)
y_pred = np.argmax(pred_prob, axis=1)

true_class=np.asarray(list(map(int, test_df['diagnosis'].tolist())))
print((y_pred == true_class).sum()/len(y_pred))

from sklearn.metrics import classification_report
print(classification_report(true_class, y_pred))

print(cohen_kappa_score(
            true_class,
            y_pred, 
            weights='quadratic'
        ))

result = ""
for pred, image in zip(y_pred, df['id_code'].to_list()):
    result += image + "\t" + str(pred) + "\n"

with open("../Output/Pred.txt", 'w+') as file:
    file.write(result)
