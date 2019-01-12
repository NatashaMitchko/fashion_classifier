# TensorFlow Fashion Classifier

Google's TensorFlow Fashion Classifier (Tutorial)[https://www.tensorflow.org/tutorials/keras/basic_classification]

This little baby model is so smart, I'm so proud of it.

Look at it knowin' what ankle booties are:

Loss per Epoch:


```
60000/60000 [==============================] - 3s 58us/step - loss: 0.5008 - acc: 0.8253
Epoch 2/5
60000/60000 [==============================] - 3s 56us/step - loss: 0.3762 - acc: 0.8645
Epoch 3/5
60000/60000 [==============================] - 3s 55us/step - loss: 0.3365 - acc: 0.8762
Epoch 4/5
60000/60000 [==============================] - 3s 56us/step - loss: 0.3137 - acc: 0.8862
Epoch 5/5
60000/60000 [==============================] - 3s 55us/step - loss: 0.2947 - acc: 0.8906
```

`model.py` defines the model, checkpoint path and class names
`train.py` trains the model and saves the weights to the checkpoint path
`trained.py` loads the checkpoint weights and does some fun stuff! 