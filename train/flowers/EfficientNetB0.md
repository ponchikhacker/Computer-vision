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
#base_path = 'dataset(imgs)/train_animals_224x224'
base_path = 'dataset(imgs)/train_flowers_224x224'
#base_path = 'dataset(imgs)/train_cars_aug_224x224'
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

    2012
    


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

    Количество изображений: 2012
    Размер обучающей выборки: 1609
    Размер валидационной выборки: 403
    (1609, 224, 224, 3)
    (1609, 2)
    (403, 224, 224, 3)
    (403, 2)
    


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

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    efficientnetb0 (Functional)  (None, 1280)              4049571   
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1280)              5120      
    _________________________________________________________________
    dense (Dense)                (None, 256)               327936    
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 4,383,141
    Trainable params: 4,338,558
    Non-trainable params: 44,583
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/15
    101/101 [==============================] - 27s 176ms/step - loss: 4.5990 - accuracy: 0.8067 - val_loss: 4.2426 - val_accuracy: 0.9454
    Epoch 2/15
    101/101 [==============================] - 15s 147ms/step - loss: 4.2173 - accuracy: 0.9204 - val_loss: 4.0650 - val_accuracy: 0.9479
    Epoch 3/15
    101/101 [==============================] - 15s 148ms/step - loss: 4.0216 - accuracy: 0.9397 - val_loss: 3.9106 - val_accuracy: 0.9553
    Epoch 4/15
    101/101 [==============================] - 16s 154ms/step - loss: 3.8826 - accuracy: 0.9366 - val_loss: 3.7636 - val_accuracy: 0.9578
    Epoch 5/15
    101/101 [==============================] - 15s 152ms/step - loss: 3.7068 - accuracy: 0.9497 - val_loss: 3.6267 - val_accuracy: 0.9653
    Epoch 6/15
    101/101 [==============================] - 15s 147ms/step - loss: 3.5553 - accuracy: 0.9646 - val_loss: 3.4915 - val_accuracy: 0.9727
    Epoch 7/15
    101/101 [==============================] - 15s 148ms/step - loss: 3.4321 - accuracy: 0.9584 - val_loss: 3.3685 - val_accuracy: 0.9702
    Epoch 8/15
    101/101 [==============================] - 15s 150ms/step - loss: 3.3175 - accuracy: 0.9608 - val_loss: 3.2524 - val_accuracy: 0.9727
    Epoch 9/15
    101/101 [==============================] - 15s 150ms/step - loss: 3.1718 - accuracy: 0.9782 - val_loss: 3.1420 - val_accuracy: 0.9826
    Epoch 10/15
    101/101 [==============================] - 16s 155ms/step - loss: 3.0495 - accuracy: 0.9820 - val_loss: 3.0314 - val_accuracy: 0.9801
    Epoch 11/15
    101/101 [==============================] - 15s 149ms/step - loss: 2.9521 - accuracy: 0.9801 - val_loss: 2.9285 - val_accuracy: 0.9727
    Epoch 12/15
    101/101 [==============================] - 15s 147ms/step - loss: 2.8438 - accuracy: 0.9820 - val_loss: 2.8331 - val_accuracy: 0.9702
    Epoch 13/15
    101/101 [==============================] - 15s 151ms/step - loss: 2.7368 - accuracy: 0.9876 - val_loss: 2.7352 - val_accuracy: 0.9727
    Epoch 14/15
    101/101 [==============================] - 15s 149ms/step - loss: 2.6598 - accuracy: 0.9814 - val_loss: 2.6413 - val_accuracy: 0.9702
    Epoch 15/15
    101/101 [==============================] - 15s 149ms/step - loss: 2.5669 - accuracy: 0.9807 - val_loss: 2.5512 - val_accuracy: 0.9702
    


```python
model.save('Model_EffB0_flowers.h5')
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


<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/84b36f0d-b832-4788-8da7-a55de8175945" />



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
test_base_path = 'dataset(imgs)/Test_flowers_aug_224x224'
#test_base_path = 'dataset(imgs)/Test_animals_aug_224x224'
#test_base_path = 'dataset(imgs)/Test_cars_aug_224x224'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(224, 224))
print(test_images.shape)
```

    (144, 224, 224, 3)
    


```python
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.9652777777777778
    {0: 'dandelion', 1: 'tulip'}
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    


```python
import matplotlib.pyplot as plt
for i in range(len(test_images)):
    plt.imshow(test_images[i])
    true_label_name = class_labels[test_labels[i]]
    predicted_label_name = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.show()
```

<img width="539" height="462" alt="output_13_4" src="https://github.com/user-attachments/assets/f5a756ce-acd5-4aeb-8333-2f8cf4c62c04" />

<img width="539" height="462" alt="output_13_10" src="https://github.com/user-attachments/assets/0afd1fec-4c43-43f9-a480-62bbe5369a6c" />

<img width="450" height="462" alt="output_13_83" src="https://github.com/user-attachments/assets/17e8e7e9-751c-44b3-afb9-3c50b44ed2d1" />

<img width="450" height="462" alt="output_13_79" src="https://github.com/user-attachments/assets/f4cdc5cb-019d-42ca-8bf2-5e88b4f28450" />


```python

```

