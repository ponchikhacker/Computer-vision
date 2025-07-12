```python
import os

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adamax
```


```python
def load_data(data_dir):
    images = []
    labels = []

    class_names = sorted(os.listdir(data_dir))
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
            labels.append(class_indices[class_name])
    
    return np.array(images), np.array(labels)
```


```python
#data_dir = 'train_flowers_224x224'
#data_dir = 'train_animals_224x224'
data_dir = 'train_cars_aug_224x224'

images, labels = load_data(data_dir)

print(len(images))
images = images / 255.0
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels_ohe = to_categorical(encoded_labels, 2)
images, encoded_labels_ohe = shuffle(images, encoded_labels_ohe, random_state=20)

x_train, x_val, y_train, y_val = train_test_split(images, encoded_labels_ohe, test_size=0.2, random_state=47)
```

    1120
    


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
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True
for layer in base_model.layers[:-3]:
    layer.trainable = False
    
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```


```python
model = Sequential([
    base_model,
    
    Flatten(),
    
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    
    Dropout(0.2),
    Dense(2, activation='softmax') # 2 класса
    
])

model.compile(optimizer=Adamax(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Functional)           (None, 7, 7, 512)         14714688  
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               6422784   
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 21,137,986
    Trainable params: 11,142,914
    Non-trainable params: 9,995,072
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/15
    56/56 [==============================] - 8s 63ms/step - loss: 5.7089 - accuracy: 0.5324 - val_loss: 5.5295 - val_accuracy: 0.5804
    Epoch 2/15
    56/56 [==============================] - 3s 59ms/step - loss: 5.4387 - accuracy: 0.6116 - val_loss: 5.3395 - val_accuracy: 0.6518
    Epoch 3/15
    56/56 [==============================] - 3s 59ms/step - loss: 5.2435 - accuracy: 0.6685 - val_loss: 5.1713 - val_accuracy: 0.6920
    Epoch 4/15
    56/56 [==============================] - 3s 59ms/step - loss: 5.0389 - accuracy: 0.7400 - val_loss: 5.0318 - val_accuracy: 0.6920
    Epoch 5/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.8811 - accuracy: 0.7779 - val_loss: 4.8785 - val_accuracy: 0.7545
    Epoch 6/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.7296 - accuracy: 0.8114 - val_loss: 4.7427 - val_accuracy: 0.7723
    Epoch 7/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.5688 - accuracy: 0.8560 - val_loss: 4.6287 - val_accuracy: 0.8080
    Epoch 8/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.4386 - accuracy: 0.8806 - val_loss: 4.5080 - val_accuracy: 0.8170
    Epoch 9/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.3226 - accuracy: 0.9018 - val_loss: 4.4216 - val_accuracy: 0.7857
    Epoch 10/15
    56/56 [==============================] - 3s 60ms/step - loss: 4.2150 - accuracy: 0.9141 - val_loss: 4.3069 - val_accuracy: 0.8304
    Epoch 11/15
    56/56 [==============================] - 3s 60ms/step - loss: 4.0992 - accuracy: 0.9297 - val_loss: 4.2235 - val_accuracy: 0.8304
    Epoch 12/15
    56/56 [==============================] - 3s 59ms/step - loss: 4.0119 - accuracy: 0.9498 - val_loss: 4.1349 - val_accuracy: 0.8304
    Epoch 13/15
    56/56 [==============================] - 3s 60ms/step - loss: 3.9127 - accuracy: 0.9609 - val_loss: 4.0861 - val_accuracy: 0.8170
    Epoch 14/15
    56/56 [==============================] - 3s 60ms/step - loss: 3.8440 - accuracy: 0.9632 - val_loss: 3.9903 - val_accuracy: 0.8348
    Epoch 15/15
    56/56 [==============================] - 3s 60ms/step - loss: 3.7597 - accuracy: 0.9743 - val_loss: 3.9199 - val_accuracy: 0.8438
    


```python
model.save('Model_VGG16_cars.h5')
```

```python
import matplotlib.pyplot as plt

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
    
<img width="1978" height="777" alt="output_8_0" src="https://github.com/user-attachments/assets/690be453-86af-4aad-bd73-9d4b6e14dd57" />

    

```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image

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
test_images, test_labels, label_map = load_images_from_folders(test_base_path, )
print(test_images.shape)
```

    (120, 224, 224, 3)
    


```python
#model = tf.keras.models.load_model('Model_VGG16_cars.h5')

predictions = model.predict(test_images)

predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.6
    {0: 'audi', 1: 'bmv'}
    [1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0 0 0 1 1 1 0
     1 0 1 1 1 1 1 0 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
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


<img width="450" height="462" alt="output_12_2" src="https://github.com/user-attachments/assets/fe1e3605-48ab-453a-a35c-69516a35fb87" />


<img width="450" height="462" alt="output_12_5" src="https://github.com/user-attachments/assets/3e8afc94-ac2b-4f5f-ad26-c783649ccb7f" />


<img width="450" height="462" alt="output_12_85" src="https://github.com/user-attachments/assets/d4443200-4e49-453d-8ef4-2a4266a7fd74" />

<img width="450" height="462" alt="output_12_119" src="https://github.com/user-attachments/assets/7a34a9a4-fe76-436b-b18b-1525c7af05a2" />


