import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Conv2D, MaxPooling2D
from pathlib import Path

(x_train,y_train), (x_test,y_test) = cifar10.load_data()


x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_train = y_train/255
y_test = y_test/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32,(3, 3), padding='same', activation='relu',input_shape= (32,32,3)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation='relu', ))
model.add(Dropout(0.50))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test,y_test),
    shuffle=True

)
model_train = model.to_json()
f = Path('model_train.json')
f.write_text(model_train)

model.save_weights('model_weights.h5')
