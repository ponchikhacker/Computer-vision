```python
import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmented_dataset(source_dir, output_dir, augmentations_per_image=5, image_size=(224, 224)):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for class_name in os.listdir(source_dir):
        source_class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(source_class_path):
            continue

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path)
        
        print(f"Обработка класса: {class_name}")

        for img_name in os.listdir(source_class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                source_img_path = os.path.join(source_class_path, img_name)
                
                try:
                    shutil.copy2(source_img_path, os.path.join(output_class_path, img_name))

                    img = Image.open(source_img_path).convert('RGB')
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, 0)

                    aug_iter = datagen.flow(
                        img_array,
                        batch_size=1,
                        save_to_dir=output_class_path,
                        save_prefix=f'aug_{os.path.splitext(img_name)[0]}',
                        save_format='jpg'
                    )

                    for i in range(augmentations_per_image):
                        next(aug_iter)

                except Exception as e:
                    print(f"Ошибка при обработке {source_img_path}: {e}")
```
```python                 
source_directory = 'Test_cars_224x224'
output_directory = 'Test_cars_aug_224x224'

create_augmented_dataset(source_directory, output_directory, augmentations_per_image=3, image_size=(224, 224))

