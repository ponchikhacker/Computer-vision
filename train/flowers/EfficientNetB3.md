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
base_path = 'train_flowers_300x300'
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
    (1609, 300, 300, 3)
    (1609, 2)
    (403, 300, 300, 3)
    (403, 2)
    


```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
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
    101/101 [==============================] - 67s 514ms/step - loss: 4.7084 - accuracy: 0.8229 - val_loss: 4.4180 - val_accuracy: 0.9082
    Epoch 2/15
    101/101 [==============================] - 47s 461ms/step - loss: 4.2867 - accuracy: 0.9267 - val_loss: 4.1384 - val_accuracy: 0.9553
    Epoch 3/15
    101/101 [==============================] - 47s 465ms/step - loss: 4.0793 - accuracy: 0.9528 - val_loss: 3.9752 - val_accuracy: 0.9603
    Epoch 4/15
    101/101 [==============================] - 47s 461ms/step - loss: 3.9000 - accuracy: 0.9664 - val_loss: 3.8272 - val_accuracy: 0.9529
    Epoch 5/15
    101/101 [==============================] - 47s 466ms/step - loss: 3.7574 - accuracy: 0.9664 - val_loss: 3.6848 - val_accuracy: 0.9628
    Epoch 6/15
    101/101 [==============================] - 48s 471ms/step - loss: 3.6044 - accuracy: 0.9751 - val_loss: 3.5507 - val_accuracy: 0.9578
    Epoch 7/15
    101/101 [==============================] - 47s 469ms/step - loss: 3.4709 - accuracy: 0.9751 - val_loss: 3.4206 - val_accuracy: 0.9578
    Epoch 8/15
    101/101 [==============================] - 47s 463ms/step - loss: 3.3355 - accuracy: 0.9807 - val_loss: 3.2957 - val_accuracy: 0.9653
    Epoch 9/15
    101/101 [==============================] - 47s 467ms/step - loss: 3.2140 - accuracy: 0.9832 - val_loss: 3.1808 - val_accuracy: 0.9702
    Epoch 10/15
    101/101 [==============================] - 47s 469ms/step - loss: 3.0822 - accuracy: 0.9913 - val_loss: 3.0646 - val_accuracy: 0.9677
    Epoch 11/15
    101/101 [==============================] - 47s 470ms/step - loss: 2.9800 - accuracy: 0.9826 - val_loss: 2.9530 - val_accuracy: 0.9727
    Epoch 12/15
    101/101 [==============================] - 51s 506ms/step - loss: 2.8522 - accuracy: 0.9956 - val_loss: 2.8518 - val_accuracy: 0.9702
    Epoch 13/15
    101/101 [==============================] - 47s 468ms/step - loss: 2.7561 - accuracy: 0.9925 - val_loss: 2.7466 - val_accuracy: 0.9727
    Epoch 14/15
    101/101 [==============================] - 47s 465ms/step - loss: 2.6588 - accuracy: 0.9907 - val_loss: 2.6508 - val_accuracy: 0.9702
    Epoch 15/15
    101/101 [==============================] - 47s 465ms/step - loss: 2.5610 - accuracy: 0.9913 - val_loss: 2.5561 - val_accuracy: 0.9727
    


```python
model.save('Model_EffB3_flowers.h5')
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

<img width="1978" height="777" alt="output_9_0" src="https://github.com/user-attachments/assets/c76626aa-beac-463b-baa8-6eb07ba9fa9f" />


```python
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
test_base_path = 'Test_flowers_aug_300x300'
#test_base_path = 'Test_animals_aug_300x300'
#test_base_path = 'Test_cars_aug_300x300'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, image_size=(300, 300))
print(test_images.shape)
```

    (144, 300, 300, 3)
    


```python
#from tensorflow.keras.models import load_model
#model = load_model('Model_EffB3_flowers.h5')

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.9930555555555556
    {0: 'dandelion', 1: 'tulip'}
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    


```python
import matplotlib.pyplot as plt
for i in range(len(test_images)):
    plt.imshow(test_images[i])
    true_label_name = class_labels[test_labels[i]]
    predicted_label_name = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.show()
```
<img width="539" height="462" alt="output_13_52" src="https://github.com/user-attachments/assets/a4f7e705-5331-4b44-988f-1668beb5262e" />


<img width="539" height="462" alt="output_13_1" src="https://github.com/user-attachments/assets/5fe1a3b6-989a-4fca-8834-788205f82b24" />


<img width="450" height="462" alt="output_13_128" src="https://github.com/user-attachments/assets/7ee32b85-5341-4865-94f2-ef1db53b3c92" />


<img width="450" height="462" alt="output_13_109" src="https://github.com/user-attachments/assets/21a30dbe-56dc-4e3b-b4b1-de8654c0ad4d" />
