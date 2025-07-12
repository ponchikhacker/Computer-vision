```python
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
```


```python
def load_images_from_folders(base_path, image_size=(224, 224)):
    data = []
    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img_array = np.array(img)
                        data.append((img_array, label))
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")

    return data
```


```python
base_path = 'train_animals_224x224'
#base_path = 'train_flowers_224x224'
#base_path = 'train_cars_aug_224x224'
data = load_images_from_folders(base_path)
```


```python
images, labels = zip(*data)

images = np.array(images)
print(len(images))
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels_ohe = to_categorical(encoded_labels, 2)
images, encoded_labels_ohe = shuffle(images, encoded_labels_ohe, random_state=20)
```

    2869
    


```python
split_index = int(0.8 * len(images))
x_train, x_val = images[:split_index], images[split_index:]
y_train, y_val = encoded_labels_ohe[:split_index], encoded_labels_ohe[split_index:]
```


```python
print("Количество изображений:", len(images))
print("Размер обучающей выборки:", len(x_train))
print("Размер валидационной выборки:", len(x_val))
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
```

    Количество изображений: 2869
    Размер обучающей выборки: 2295
    Размер валидационной выборки: 574
    (2295, 224, 224, 3)
    (2295, 2)
    (574, 224, 224, 3)
    (574, 2)
    


```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# NN with EfficientNetB3
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="max"
)

model = Sequential([
    base_model,
    BatchNormalization(),
    
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    
    Dense(2, activation='softmax') # 2 класса
])

model.compile(optimizer=Adamax(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    efficientnetb0 (Functional)  (None, 1280)              4049571   
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 1280)              5120      
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               327936    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 4,383,141
    Trainable params: 4,338,558
    Non-trainable params: 44,583
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/15
    144/144 [==============================] - 33s 166ms/step - loss: 4.4709 - accuracy: 0.8471 - val_loss: 4.0991 - val_accuracy: 0.9564
    Epoch 2/15
    144/144 [==============================] - 21s 146ms/step - loss: 4.0237 - accuracy: 0.9325 - val_loss: 3.8421 - val_accuracy: 0.9582
    Epoch 3/15
    144/144 [==============================] - 21s 147ms/step - loss: 3.7442 - accuracy: 0.9547 - val_loss: 3.5939 - val_accuracy: 0.9669
    Epoch 4/15
    144/144 [==============================] - 21s 148ms/step - loss: 3.5086 - accuracy: 0.9625 - val_loss: 3.3807 - val_accuracy: 0.9739
    Epoch 5/15
    144/144 [==============================] - 22s 151ms/step - loss: 3.2883 - accuracy: 0.9739 - val_loss: 3.1837 - val_accuracy: 0.9756
    Epoch 6/15
    144/144 [==============================] - 21s 147ms/step - loss: 3.1045 - accuracy: 0.9739 - val_loss: 3.0008 - val_accuracy: 0.9721
    Epoch 7/15
    144/144 [==============================] - 22s 150ms/step - loss: 2.9273 - accuracy: 0.9756 - val_loss: 2.8392 - val_accuracy: 0.9756
    Epoch 8/15
    144/144 [==============================] - 21s 147ms/step - loss: 2.7698 - accuracy: 0.9765 - val_loss: 2.6910 - val_accuracy: 0.9791
    Epoch 9/15
    144/144 [==============================] - 21s 149ms/step - loss: 2.6096 - accuracy: 0.9887 - val_loss: 2.5530 - val_accuracy: 0.9774
    Epoch 10/15
    144/144 [==============================] - 21s 149ms/step - loss: 2.4840 - accuracy: 0.9865 - val_loss: 2.4254 - val_accuracy: 0.9808
    Epoch 11/15
    144/144 [==============================] - 21s 148ms/step - loss: 2.3571 - accuracy: 0.9874 - val_loss: 2.3095 - val_accuracy: 0.9791
    Epoch 12/15
    144/144 [==============================] - 21s 149ms/step - loss: 2.2427 - accuracy: 0.9878 - val_loss: 2.1948 - val_accuracy: 0.9826
    Epoch 13/15
    144/144 [==============================] - 21s 149ms/step - loss: 2.1287 - accuracy: 0.9908 - val_loss: 2.0856 - val_accuracy: 0.9826
    Epoch 14/15
    144/144 [==============================] - 21s 149ms/step - loss: 2.0219 - accuracy: 0.9917 - val_loss: 1.9802 - val_accuracy: 0.9826
    Epoch 15/15
    144/144 [==============================] - 22s 149ms/step - loss: 1.9252 - accuracy: 0.9887 - val_loss: 1.8868 - val_accuracy: 0.9843
    


```python
model.save('Model_EffB0_animals.h5')
```
```python
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

Epochs = [i+1 for i in range(len(tr_acc))]

plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'orange', label= 'Training loss')
plt.plot(Epochs, val_loss, label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'red', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'orange', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'red', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```


    
<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/f618ad96-bb25-4211-a5cb-4d9ee24590e8" />



```python
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_folders(base_path, image_size=(224, 224)):
    images = []
    labels = []
    label_map = {}
    current_label_index = 0

    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        
        if os.path.isdir(folder_path):
            if label not in label_map:
                label_map[label] = current_label_index
                current_label_index += 1
                
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
               
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(label_map[label])
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_map
```


```python
#test_base_path = 'Test_flowers_aug_224x224'
test_base_path = 'Test_animals_aug_224x224'
#test_base_path = 'Test_cars_aug_224x224'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(224, 224))
print(test_images.shape)
```

    (240, 224, 224, 3)
    


```python
#model = tf.keras.models.load_model('Model_EffB0_animals.h5')
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)
```

    Test accuracy: 0.9916666666666667
    {0: 'cats', 1: 'dogs'}
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    


```python
import matplotlib.pyplot as plt
for i in range(len(test_images)):
    plt.imshow(test_images[i])
    true_label_name = class_labels[test_labels[i]]
    predicted_label_name = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.show()
```

    

<img width="450" height="462" alt="output_13_0" src="https://github.com/user-attachments/assets/ba99a933-e875-44fd-a91a-d69e06ea309a" />

    

<img width="450" height="462" alt="output_13_84" src="https://github.com/user-attachments/assets/31d8ccb2-493f-409e-b7b7-61a777145b1d" />



<img width="450" height="462" alt="output_13_120" src="https://github.com/user-attachments/assets/f296e3fd-11b9-4e13-9f0c-e476ec8a0254" />



<img width="450" height="462" alt="output_13_149" src="https://github.com/user-attachments/assets/25d5a130-65e8-41e2-ba04-77bc59bc9512" />
