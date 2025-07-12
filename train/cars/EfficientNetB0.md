```python
import os
import numpy as np 
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
#base_path = 'train_animals_224x224'
#base_path = 'train_flowers_224x224'
base_path = 'train_cars_aug_224x224'
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

    1120
    


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

    Количество изображений: 1120
    Размер обучающей выборки: 896
    Размер валидационной выборки: 224
    (896, 224, 224, 3)
    (896, 2)
    (224, 224, 224, 3)
    (224, 2)
    


```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# NN with EfficientNetB0
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
    56/56 [==============================] - 17s 155ms/step - loss: 5.0101 - accuracy: 0.6116 - val_loss: 4.8427 - val_accuracy: 0.6696
    Epoch 2/15
    56/56 [==============================] - 8s 140ms/step - loss: 4.6271 - accuracy: 0.7723 - val_loss: 4.5975 - val_accuracy: 0.7902
    Epoch 3/15
    56/56 [==============================] - 8s 140ms/step - loss: 4.4412 - accuracy: 0.8426 - val_loss: 4.3902 - val_accuracy: 0.8527
    Epoch 4/15
    56/56 [==============================] - 8s 142ms/step - loss: 4.3105 - accuracy: 0.8583 - val_loss: 4.2460 - val_accuracy: 0.9152
    Epoch 5/15
    56/56 [==============================] - 8s 141ms/step - loss: 4.1250 - accuracy: 0.9230 - val_loss: 4.1240 - val_accuracy: 0.9196
    Epoch 6/15
    56/56 [==============================] - 8s 141ms/step - loss: 4.0302 - accuracy: 0.9263 - val_loss: 4.0203 - val_accuracy: 0.9286
    Epoch 7/15
    56/56 [==============================] - 8s 140ms/step - loss: 3.9117 - accuracy: 0.9487 - val_loss: 3.9128 - val_accuracy: 0.9509
    Epoch 8/15
    56/56 [==============================] - 8s 140ms/step - loss: 3.8275 - accuracy: 0.9587 - val_loss: 3.8076 - val_accuracy: 0.9509
    Epoch 9/15
    56/56 [==============================] - 8s 142ms/step - loss: 3.7373 - accuracy: 0.9509 - val_loss: 3.7198 - val_accuracy: 0.9509
    Epoch 10/15
    56/56 [==============================] - 8s 140ms/step - loss: 3.6318 - accuracy: 0.9710 - val_loss: 3.6291 - val_accuracy: 0.9598
    Epoch 11/15
    56/56 [==============================] - 8s 142ms/step - loss: 3.5368 - accuracy: 0.9766 - val_loss: 3.5329 - val_accuracy: 0.9643
    Epoch 12/15
    56/56 [==============================] - 8s 140ms/step - loss: 3.4786 - accuracy: 0.9688 - val_loss: 3.4461 - val_accuracy: 0.9777
    Epoch 13/15
    56/56 [==============================] - 8s 140ms/step - loss: 3.3706 - accuracy: 0.9855 - val_loss: 3.3671 - val_accuracy: 0.9732
    Epoch 14/15
    56/56 [==============================] - 8s 141ms/step - loss: 3.3109 - accuracy: 0.9777 - val_loss: 3.2817 - val_accuracy: 0.9821
    Epoch 15/15
    56/56 [==============================] - 8s 141ms/step - loss: 3.2249 - accuracy: 0.9799 - val_loss: 3.1998 - val_accuracy: 0.9821
    


```python
model.save('Model_EffB0_cars.h5')
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

<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/3436e630-7678-4984-a386-06a73e977c8a" />


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
#test_base_path = 'Test_animals_aug_224x224'
test_base_path = 'Test_cars_aug_224x224'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(224, 224))
print(test_images.shape)
```

    (120, 224, 224, 3)
    


```python
#model = tf.keras.models.load_model('Model_EffB0_cars.h5')
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)
```

    Test accuracy: 0.8333333333333334
    {0: 'audi', 1: 'bmv'}
    [0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 0 1 0 0
     0 0 1 1 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 0 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1]
    


```python
import matplotlib.pyplot as plt
for i in range(len(test_images)):
    plt.imshow(test_images[i])
    true_label_name = class_labels[test_labels[i]]
    predicted_label_name = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.show()
```

<img width="450" height="462" alt="output_13_13" src="https://github.com/user-attachments/assets/9e9867ff-a5c7-437a-9ad2-534589739ae3" />


<img width="450" height="462" alt="output_13_1" src="https://github.com/user-attachments/assets/481e810f-5b72-4675-a4d0-29c5bd939246" />


<img width="450" height="462" alt="output_13_85" src="https://github.com/user-attachments/assets/021a5f98-a747-4307-8424-73faa8e2844a" />


<img width="450" height="462" alt="output_13_81" src="https://github.com/user-attachments/assets/59173656-8275-43ef-9216-23fae3c2dce4" />
