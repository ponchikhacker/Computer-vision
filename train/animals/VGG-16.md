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
#data_dir = 'dataset(imgs)/train_flowers_224x224'
data_dir = 'dataset(imgs)/train_animals_224x224'
#data_dir = 'dataset(imgs)/train_cars_aug_224x224'

images, labels = load_data(data_dir)

print(len(images))
images = images / 255.0
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels_ohe = to_categorical(encoded_labels, 2)
images, encoded_labels_ohe = shuffle(images, encoded_labels_ohe, random_state=20)

x_train, x_val, y_train, y_val = train_test_split(images, encoded_labels_ohe, test_size=0.2, random_state=47)
```

    2869
    


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
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True
for layer in base_model.layers[:-2]:
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

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Functional)           (None, 7, 7, 512)         14714688  
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 25088)             0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               6422784   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 21,137,986
    Trainable params: 8,783,106
    Non-trainable params: 12,354,880
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=12, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/12
    144/144 [==============================] - 11s 74ms/step - loss: 5.4180 - accuracy: 0.6924 - val_loss: 5.1129 - val_accuracy: 0.8223
    Epoch 2/12
    144/144 [==============================] - 8s 58ms/step - loss: 4.9425 - accuracy: 0.8122 - val_loss: 4.7517 - val_accuracy: 0.8641
    Epoch 3/12
    144/144 [==============================] - 8s 58ms/step - loss: 4.6243 - accuracy: 0.8436 - val_loss: 4.4802 - val_accuracy: 0.8833
    Epoch 4/12
    144/144 [==============================] - 8s 58ms/step - loss: 4.3581 - accuracy: 0.8841 - val_loss: 4.2494 - val_accuracy: 0.8920
    Epoch 5/12
    144/144 [==============================] - 8s 58ms/step - loss: 4.1324 - accuracy: 0.9028 - val_loss: 4.0556 - val_accuracy: 0.9094
    Epoch 6/12
    144/144 [==============================] - 8s 58ms/step - loss: 3.9465 - accuracy: 0.9115 - val_loss: 3.8897 - val_accuracy: 0.9146
    Epoch 7/12
    144/144 [==============================] - 8s 58ms/step - loss: 3.7931 - accuracy: 0.9203 - val_loss: 3.7499 - val_accuracy: 0.9111
    Epoch 8/12
    144/144 [==============================] - 8s 58ms/step - loss: 3.6431 - accuracy: 0.9333 - val_loss: 3.6237 - val_accuracy: 0.9268
    Epoch 9/12
    144/144 [==============================] - 8s 58ms/step - loss: 3.5205 - accuracy: 0.9420 - val_loss: 3.5155 - val_accuracy: 0.9251
    Epoch 10/12
    144/144 [==============================] - 8s 58ms/step - loss: 3.4097 - accuracy: 0.9499 - val_loss: 3.4127 - val_accuracy: 0.9216
    Epoch 11/12
    144/144 [==============================] - 8s 59ms/step - loss: 3.3049 - accuracy: 0.9542 - val_loss: 3.3206 - val_accuracy: 0.9268
    Epoch 12/12
    144/144 [==============================] - 8s 59ms/step - loss: 3.2121 - accuracy: 0.9612 - val_loss: 3.2353 - val_accuracy: 0.9251
    


```python
model.save('Model_VGG16_animals.h5')
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

 <img width="1978" height="777" alt="output_8_0" src="https://github.com/user-attachments/assets/023cd1df-fca5-4480-8626-34210c6e482d" />

 
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
test_base_path = 'Test_animals_aug_224x224'
#test_base_path = 'Test_cars_aug_224x224'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, )
print(test_images.shape)
```

    (240, 224, 224, 3) (240,)
    


```python
#model = tf.keras.models.load_model('Model_8.0_VGG.h5')

predictions = model.predict(test_images)

predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.9375
    {0: 'cats', 1: 'dogs'}
    [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     1 0 0 1 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1
     1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
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

<img width="450" height="462" alt="output_12_22" src="https://github.com/user-attachments/assets/613662a5-19d4-4f14-9c68-ed293e742d4c" />


<img width="450" height="462" alt="output_12_45" src="https://github.com/user-attachments/assets/693ae81f-4d34-4589-9f94-7c5ad63103d4" />


<img width="450" height="462" alt="output_12_144" src="https://github.com/user-attachments/assets/ada33040-4cbb-4285-ab1d-4245d5c5b660" />

<img width="450" height="462" alt="output_12_160" src="https://github.com/user-attachments/assets/89826750-2e66-4112-a443-b42e938ab550" />
