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
from sklearn.metrics import confusion_matrix,classification_report

import tensorflow as tf
from tensorflow import keras
```


```python
def load_images_from_folders(base_path, image_size=(300, 300)):
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
#base_path = 'train_animals_300x300'
#base_path = 'train_flowers_300x300'
base_path = 'train_cars_aug_300x300'
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
    (896, 300, 300, 3)
    (896, 2)
    (224, 300, 300, 3)
    (224, 2)
    


```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# NN with EfficientNetB3
base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(300, 300, 3),
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

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    efficientnetb3 (Functional)  (None, 1536)              10783535  
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 1536)              6144      
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               393472    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 11,183,665
    Trainable params: 11,093,290
    Non-trainable params: 90,375
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/15
    56/56 [==============================] - 33s 467ms/step - loss: 5.1076 - accuracy: 0.5982 - val_loss: 5.0519 - val_accuracy: 0.6562
    Epoch 2/15
    56/56 [==============================] - 25s 442ms/step - loss: 4.6306 - accuracy: 0.8125 - val_loss: 4.5610 - val_accuracy: 0.8259
    Epoch 3/15
    56/56 [==============================] - 25s 443ms/step - loss: 4.3896 - accuracy: 0.8884 - val_loss: 4.3229 - val_accuracy: 0.9152
    Epoch 4/15
    56/56 [==============================] - 25s 445ms/step - loss: 4.2250 - accuracy: 0.9397 - val_loss: 4.1600 - val_accuracy: 0.9554
    Epoch 5/15
    56/56 [==============================] - 25s 446ms/step - loss: 4.0991 - accuracy: 0.9464 - val_loss: 4.0335 - val_accuracy: 0.9643
    Epoch 6/15
    56/56 [==============================] - 25s 448ms/step - loss: 3.9658 - accuracy: 0.9676 - val_loss: 3.9186 - val_accuracy: 0.9732
    Epoch 7/15
    56/56 [==============================] - 25s 453ms/step - loss: 3.8557 - accuracy: 0.9766 - val_loss: 3.8147 - val_accuracy: 0.9777
    Epoch 8/15
    56/56 [==============================] - 26s 460ms/step - loss: 3.7570 - accuracy: 0.9777 - val_loss: 3.7175 - val_accuracy: 0.9821
    Epoch 9/15
    56/56 [==============================] - 26s 459ms/step - loss: 3.6550 - accuracy: 0.9844 - val_loss: 3.6256 - val_accuracy: 0.9821
    Epoch 10/15
    56/56 [==============================] - 25s 453ms/step - loss: 3.5731 - accuracy: 0.9810 - val_loss: 3.5332 - val_accuracy: 0.9866
    Epoch 11/15
    56/56 [==============================] - 25s 452ms/step - loss: 3.4682 - accuracy: 0.9888 - val_loss: 3.4406 - val_accuracy: 0.9866
    Epoch 12/15
    56/56 [==============================] - 25s 451ms/step - loss: 3.3829 - accuracy: 0.9900 - val_loss: 3.3535 - val_accuracy: 0.9866
    Epoch 13/15
    56/56 [==============================] - 25s 446ms/step - loss: 3.2950 - accuracy: 0.9933 - val_loss: 3.2672 - val_accuracy: 0.9866
    Epoch 14/15
    56/56 [==============================] - 25s 450ms/step - loss: 3.2112 - accuracy: 0.9944 - val_loss: 3.1834 - val_accuracy: 0.9911
    Epoch 15/15
    56/56 [==============================] - 25s 451ms/step - loss: 3.1278 - accuracy: 0.9933 - val_loss: 3.0960 - val_accuracy: 0.9911
    


```python
model.save('Model_EffB3_cars.h5')
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

<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/f80e2274-abbe-435d-9518-fc64cfb265cf" />


```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def load_images_from_folders(base_path, image_size=(300, 300)):
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
#test_base_path = 'Test_flowers_aug_300x300'
#test_base_path = 'Test_animals_aug_300x300'
test_base_path = 'Test_cars_aug_300x300'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(300, 300))
print(test_images.shape)
```

    (120, 300, 300, 3)
    


```python
#from tensorflow.keras.models import load_model
#model = load_model('Model_EffB3_cars.h5')

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.8583333333333333
    {0: 'audi', 1: 'bmv'}
    [0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1
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

<img width="450" height="462" alt="output_13_2" src="https://github.com/user-attachments/assets/f3e4b9e0-2114-4da7-aeef-e35967c2b1ed" />

<img width="450" height="462" alt="output_13_14" src="https://github.com/user-attachments/assets/faa4e576-c314-4c7d-b32b-c53c3b566bbc" />


<img width="450" height="462" alt="output_13_87" src="https://github.com/user-attachments/assets/1f4e23c0-c309-42e6-94a9-ece458650d33" />

<img width="450" height="462" alt="output_13_104" src="https://github.com/user-attachments/assets/493a3d4d-631d-4d49-9542-c5f5290ba38e" />
