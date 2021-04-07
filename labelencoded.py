#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


# In[10]:


morgan_train = pd.read_csv('data_/morgan_0_train.csv')
morgan_test = pd.read_csv('data_/morgan_0_test.csv')


# In[11]:

cell_train2 = pd.read_csv('default/binarized_cells_0_train.csv')
cell_test2 = pd.read_csv('default/binarized_cells_0_test.csv')

from sklearn.preprocessing import StandardScaler

cell_train = pd.read_csv('data_/pcad_cells_0_train.csv')
cell_test = pd.read_csv('data_/pcad_cells_0_test.csv')

scaler = StandardScaler()
scaler.fit(cell_train)
cell_train = pd.DataFrame(scaler.transform(cell_train))
cell_test = pd.DataFrame(scaler.transform(cell_test))


# In[12]:


X_train = pd.concat((morgan_train, cell_train, cell_train2), axis=1)
X_test = pd.concat((morgan_test, cell_test, cell_test2), axis=1)


# In[13]:


y_train = pd.read_csv('data_/y_train_0.csv')
y_test = pd.read_csv('data_/y_test_0.csv')


# In[14]:


from sklearn.metrics import r2_score


# In[15]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[16]:
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    _ = Dense(units=2048, activation='relu')(input_layer)
    _ = Dropout(0.25)(_)
    _ = Dense(units=1024, activation='relu')(_)
    _ = Dropout(0.2)(_)
    _ = Dense(units=512, activation='relu')(_)
    _ = Dropout(0.15)(_)
    _ = Dense(units=256, activation='relu')(_)
    _ = Dropout(0.1)(_)
    _ = Dense(units=128, activation='relu')(_)
    _ = Dropout(0.05)(_)
    # _ = BatchNormalization()(_)
    # _ = Dropout(0.25)(_)
    # _ = Dense(units=256, activation='relu')(_)
    # _ = BatchNormalization()(_)
    # _ = Dense(units=512, activation='relu')(_)
    # _ = BatchNormalization()(_)
    # _ = Dense(units=128, activation='relu')(_)
    # _ = Dense(units=64, activation='relu')(_)
    # _ = Dropout(0.2)(_)
    # _ = Dense(units=128)(_)
    # _ = BatchNormalization()(_)
    # _ = Activation('relu')(_)
    # _ = Dropout(0.1)(_)
    output_layer = Dense(units=1, activation='linear')(_)

    model = Model(inputs=input_layer, outputs=output_layer)
    opt = Adam(learning_rate=0.0004)
    model.compile(optimizer=opt, loss = root_mean_squared_error)

    return model


# In[22]:

#
# import tensorflow as tf
# tf.test.gpu_device_name()

# In[17]:


model = build_model(X_train.shape[1])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=80, verbose=1)

y_preds = model.predict(X_test)
print(r2_score(y_test, y_preds))


# In[50]:


from sklearn.metrics import mean_squared_error
from math import sqrt
# from sklearn.linear_model import Ridge
# regressor = Ridge(alpha=50, max_iter=10000)
# regressor.fit(X_train, y_train)
# y_preds = regressor.predict(X_test)
# print(r2_score(y_test, y_preds))
print(sqrt(mean_squared_error(y_test, y_preds)))


# In[37]:


# model = build_model(X_train.shape[1])
# model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=100, verbose=1)
#
# y_preds = model.predict(X_test)
# print(r2_score(y_test, y_preds))


# In[38]:



# In[40]:


model.save("paper")
