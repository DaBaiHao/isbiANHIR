from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard

model=Sequential([
    Dense(8, init='uniform', activation='relu'),
    Dense(1,activation='sigmoid')
])