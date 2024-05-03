from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

train_data_preprocess = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#rescale refers to multiplying 1./255 to all pixel to bring the values int range of 0 to 1 it does not change the dimension of the image
#shear_range refers to the rotation or moving an image in any angle or direction in real life sceanrio the images can be tlited or moved so this can help in better learning. the 0.2 represent 20% allowance

val_data_preprocess = ImageDataGenerator(rescale=1./255)

training_data = train_data_preprocess.flow_from_directory(r"D:\Pantech ai\basic pgm in keras & tensorflow\image classification\dataset\train",
                                                          target_size=(64,64),
                                                          batch_size=8,
                                                          class_mode='binary')

val_data = val_data_preprocess.flow_from_directory(r"D:\Pantech ai\basic pgm in keras & tensorflow\image classification\dataset\val",
                                                   target_size=(64,64),
                                                   batch_size=8,
                                                   class_mode='binary')

model.fit_generator(training_data,steps_per_epoch = 10, epochs = 25, validation_data = val_data, validation_steps = 2)

model_json = model.to_json()
with open("model.json","w")  as file:
    file.write(model_json)
model.save_weights("model.h5")

