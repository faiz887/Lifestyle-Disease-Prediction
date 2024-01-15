import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

image_directory = 'chest_xray/test'
no_tumor_images = os.listdir(os.path.join(image_directory, 'NORMAL'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'PNEUMONIA'))

dataset = []
label = []
INPUT_SIZE = 64

for image_name in no_tumor_images:
    if image_name.lower().endswith('.jpeg'):
        image_path = os.path.join(image_directory, 'NORMAL', image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)
        else:
            print(f"Failed to read: {image_path}")

for image_name in yes_tumor_images:
    if image_name.lower().endswith('.jpeg'):
        image_path = os.path.join(image_directory, 'PNEUMONIA', image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)
        else:
            print(f"Failed to read: {image_path}")

dataset = np.array(dataset)
label = np.array(label)

# Check if the dataset is not empty before splitting
if len(dataset) > 0:
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    # ... (previous code)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ... (add more Conv2D and MaxPooling2D layers as needed)

    model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2))  # Change the number of nodes to 2 for binary classification
    model.add(Activation('softmax'))  # Change activation to 'softmax'

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=True)
    model.save("BrainTumor10_epochs.h5py")


else:
    print("Empty dataset. Check the image_directory path.")




