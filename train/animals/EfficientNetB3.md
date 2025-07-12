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
base_path = 'train_animals_300x300'
#base_path = 'train_flowers_300x300'
#base_path = 'train_cars_aug_300x300'
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
    (2295, 300, 300, 3)
    (2295, 2)
    (574, 300, 300, 3)
    (574, 2)
    


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

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    efficientnetb3 (Functional)  (None, 1536)              10783535  
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1536)              6144      
    _________________________________________________________________
    dense (Dense)                (None, 256)               393472    
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 11,183,665
    Trainable params: 11,093,290
    Non-trainable params: 90,375
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/15
    144/144 [==============================] - 98s 493ms/step - loss: 4.5316 - accuracy: 0.8867 - val_loss: 4.1930 - val_accuracy: 0.9826
    Epoch 2/15
    144/144 [==============================] - 66s 461ms/step - loss: 4.1465 - accuracy: 0.9608 - val_loss: 3.9712 - val_accuracy: 0.9791
    Epoch 3/15
    144/144 [==============================] - 66s 462ms/step - loss: 3.9337 - accuracy: 0.9630 - val_loss: 3.7790 - val_accuracy: 0.9843
    Epoch 4/15
    144/144 [==============================] - 66s 461ms/step - loss: 3.7147 - accuracy: 0.9734 - val_loss: 3.6000 - val_accuracy: 0.9878
    Epoch 5/15
    144/144 [==============================] - 66s 460ms/step - loss: 3.5206 - accuracy: 0.9847 - val_loss: 3.4251 - val_accuracy: 0.9878
    Epoch 6/15
    144/144 [==============================] - 66s 461ms/step - loss: 3.3634 - accuracy: 0.9804 - val_loss: 3.2599 - val_accuracy: 0.9861
    Epoch 7/15
    144/144 [==============================] - 67s 466ms/step - loss: 3.1872 - accuracy: 0.9852 - val_loss: 3.0968 - val_accuracy: 0.9895
    Epoch 8/15
    144/144 [==============================] - 67s 467ms/step - loss: 3.0444 - accuracy: 0.9856 - val_loss: 2.9659 - val_accuracy: 0.9895
    Epoch 9/15
    144/144 [==============================] - 67s 468ms/step - loss: 2.9068 - accuracy: 0.9904 - val_loss: 2.8377 - val_accuracy: 0.9895
    Epoch 10/15
    144/144 [==============================] - 67s 463ms/step - loss: 2.7734 - accuracy: 0.9895 - val_loss: 2.7061 - val_accuracy: 0.9895
    Epoch 11/15
    144/144 [==============================] - 67s 463ms/step - loss: 2.6456 - accuracy: 0.9900 - val_loss: 2.5836 - val_accuracy: 0.9878
    Epoch 12/15
    144/144 [==============================] - 67s 464ms/step - loss: 2.5258 - accuracy: 0.9908 - val_loss: 2.4623 - val_accuracy: 0.9895
    Epoch 13/15
    144/144 [==============================] - 67s 465ms/step - loss: 2.3977 - accuracy: 0.9935 - val_loss: 2.3419 - val_accuracy: 0.9913
    Epoch 14/15
    144/144 [==============================] - 67s 463ms/step - loss: 2.2729 - accuracy: 0.9965 - val_loss: 2.2189 - val_accuracy: 0.9913
    Epoch 15/15
    144/144 [==============================] - 67s 463ms/step - loss: 2.1516 - accuracy: 0.9956 - val_loss: 2.0962 - val_accuracy: 0.9895
    


```python
model.save('Model_EffB3_animals.h5')
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


<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/72bd67d7-7e38-438e-bf8a-62b36f9cb1c4" />

    

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
test_base_path = 'Test_animals_aug_300x300'
#test_base_path = 'Test_cars_aug_300x300'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(300, 300))
print(test_images.shape)
```

    (240, 300, 300, 3)
    


```python
#from tensorflow.keras.models import load_model
#model = load_model('Model_7.0_Eff.h5')

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.9958333333333333
    {0: 'cats', 1: 'dogs'}
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
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

<img width="450" height="462" alt="output_13_9" src="https://github.com/user-attachments/assets/cb43191d-1b6a-4f37-8131-4b6d8835889f" />

<img width="450" height="462" alt="output_13_19" src="https://github.com/user-attachments/assets/11d66e23-a0cf-44fa-b0c3-5bfae995d250" />

<img width="450" height="462" alt="output_13_133" src="https://github.com/user-attachments/assets/60b87383-faa0-4a03-914d-f5b49a92d6f7" />

<img width="450" height="462" alt="output_13_140" src="https://github.com/user-attachments/assets/eb76d2ef-0298-46d5-b5f1-80093f416be4" />
