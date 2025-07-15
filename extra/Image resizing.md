```python
import os
import numpy as np
from PIL import Image

def resize_with_padding(image, target_size=(224, 224)):
    original_size = image.size
    ratio = float(target_size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", target_size)
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_image

def load_and_save_data(data_dir, output_dir):
    images = []
    labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = sorted(os.listdir(data_dir))
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            img = Image.open(img_path)
            img = resize_with_padding(img)
            
            output_fname = os.path.join(output_class_dir, fname)
            img.save(output_fname)
```
```python
data_dir = 'Test_cars'
output_dir = 'Test_cars_224x224'
load_and_save_data(data_dir, output_dir)

```
