import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt
from model import create_model, checkpoint_path, class_names

print("TensorFlow version: ", tf.__version__)

# import the fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
print(len(train_labels))

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show() 

# preprocess images by normalizing pixel values (0-225) -> (0-1)
train_images = train_images / 255
test_images = test_images / 255

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
#                                                  save_weights_only=True,
#                                                  verbose=1)

model = create_model()

# Time to train the model
history = model.fit(train_images, train_labels, epochs=5)
model.save_weights(checkpoint_path)

# Plot the loss after each epoch
print(history)
plt.plot(range(len(history.history["loss"])), history.history["loss"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])
predicted_label = class_names[np.argmax(predictions[0])]

# Show the first image with it's predicted label
plt.figure()
plt.imshow(train_images[0])
plt.xlabel(predicted_label)
plt.grid(False)
plt.show() 

