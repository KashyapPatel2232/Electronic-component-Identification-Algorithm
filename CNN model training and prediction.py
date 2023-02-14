import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the directory for all component images
all_images_dir = "D:\\DRDO (Dare to dream 4.0 competition)\\Electronic component data set\\images"

# Create dictionaries to store the images and labels for each component
component_images = {}
component_labels = {}

# Loop through each subdirectory in the all_images_dir directory
for component_dir in os.listdir(all_images_dir):

    # Set the full path for the component subdirectory
    component_dir_path = os.path.join(all_images_dir, component_dir)

    # Skip any files that are not directories
    if not os.path.isdir(component_dir_path):
        continue

    # Initialize the lists for the images and labels for this component
    component_images[component_dir] = []
    component_labels[component_dir] = []

    # Loop through each image in the component subdirectory
    for image_file in os.listdir(component_dir_path):

        # Set the full path for the image
        image_path = os.path.join(component_dir_path, image_file)

        # Read in the image and append it to the list of images for this component
        image = cv2.imread(image_path)
        component_images[component_dir].append(image)

        # Append the label for this image to the list of labels for this component
        component_labels[component_dir].append(component_dir)

# Combine all the component images and labels into single lists
images = []
labels = []
for component in component_images.keys():
    images.extend(component_images[component])
    labels.extend(component_labels[component])

# Convert the images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Perform one-hot encoding on the labels
labels = to_categorical(labels)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=train_images[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(component_images.keys()), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

# Load the testing images and labels
testing_dir = '"D:\\DRDO (Dare to dream 4.0 competition)\\Electronic component data set\\test"'
testing_images = []
testing_labels = []
for component_dir in os.listdir(testing_dir):
    component_dir_path = os.path.join(testing_dir, component_dir)
    if not os.path.isdir(component_dir_path):
        continue
    for image_file in os.listdir(component_dir_path):
        image_path = os.path.join(component_dir_path, image_file)
        image = cv2.imread(image_path)
        testing_images.append(image)
        testing_labels

print(testing_labels)
