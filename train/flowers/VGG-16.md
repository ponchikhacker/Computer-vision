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
data_dir = 'dataset(imgs)/train_flowers_224x224'
#data_dir = 'dataset(imgs)/train_animals_224x224'
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

    2012
    


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
    Trainable params: 8,783,106
    Non-trainable params: 12,354,880
    _________________________________________________________________
    


```python
history = model.fit(x_train, y_train, epochs=12, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
```

    Epoch 1/12
    101/101 [==============================] - 32s 78ms/step - loss: 5.4181 - accuracy: 0.7290 - val_loss: 5.1084 - val_accuracy: 0.8734
    Epoch 2/12
    101/101 [==============================] - 6s 59ms/step - loss: 4.9057 - accuracy: 0.8943 - val_loss: 4.7397 - val_accuracy: 0.9007
    Epoch 3/12
    101/101 [==============================] - 6s 58ms/step - loss: 4.5773 - accuracy: 0.9124 - val_loss: 4.4523 - val_accuracy: 0.9156
    Epoch 4/12
    101/101 [==============================] - 6s 58ms/step - loss: 4.3024 - accuracy: 0.9236 - val_loss: 4.2114 - val_accuracy: 0.9256
    Epoch 5/12
    101/101 [==============================] - 6s 58ms/step - loss: 4.0745 - accuracy: 0.9397 - val_loss: 4.0047 - val_accuracy: 0.9256
    Epoch 6/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.8771 - accuracy: 0.9428 - val_loss: 3.8270 - val_accuracy: 0.9256
    Epoch 7/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.6967 - accuracy: 0.9490 - val_loss: 3.6655 - val_accuracy: 0.9380
    Epoch 8/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.5445 - accuracy: 0.9528 - val_loss: 3.5241 - val_accuracy: 0.9305
    Epoch 9/12
    101/101 [==============================] - 6s 59ms/step - loss: 3.4055 - accuracy: 0.9590 - val_loss: 3.4078 - val_accuracy: 0.9305
    Epoch 10/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.2857 - accuracy: 0.9627 - val_loss: 3.2898 - val_accuracy: 0.9380
    Epoch 11/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.1702 - accuracy: 0.9664 - val_loss: 3.1860 - val_accuracy: 0.9429
    Epoch 12/12
    101/101 [==============================] - 6s 58ms/step - loss: 3.0719 - accuracy: 0.9708 - val_loss: 3.0949 - val_accuracy: 0.9454
    


```python
model.save('Model_VGG16_flowers.h5')
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
<img width="1978" height="777" alt="output_8_0" src="https://github.com/user-attachments/assets/508fda8b-0ca3-4505-b99f-570f5de590d1" />


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
test_base_path = 'dataset(imgs)/Test_flowers_aug_224x224'
#test_base_path = 'dataset(imgs)/Test_animals_aug_224x224'
#test_base_path = 'dataset(imgs)/Test_cars_aug_224x224'
test_images, test_labels, label_map = load_images_from_folders(test_base_path, )
print(test_images.shape, test_labels.shape)
```

    (144, 224, 224, 3) (144,)
    


```python
#model = tf.keras.models.load_model('Model_VGG16_flowers.h5')

predictions = model.predict(test_images)

predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == test_labels)
print('Test accuracy:', accuracy)

class_labels = {v: k for k, v in label_map.items()}

print(class_labels)
print(predicted_classes)

```

    Test accuracy: 0.9236111111111112
    {0: 'dandelion', 1: 'tulip'}
    [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0
     0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    


```python
import matplotlib.pyplot as plt
for i in range(len(test_images)):
    plt.imshow(test_images[i])
    true_label_name = class_labels[test_labels[i]]
    predicted_label_name = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.show()
```
<img width="478" height="462" alt="output_12_6" src="https://github.com/user-attachments/assets/03230aba-776c-4535-b135-843506f18984" />

<img width="478" height="462" alt="output_12_68" src="https://github.com/user-attachments/assets/83707c6a-5628-4ecd-bef6-8776213cadc9" />

<img width="450" height="462" alt="output_12_139" src="https://github.com/user-attachments/assets/750f7f1d-3679-48be-9d60-48b8af6685ae" />

<img width="450" height="462" alt="output_12_131" src="https://github.com/user-attachments/assets/ca6ff382-d219-4795-812c-6a8427c20c44" />

