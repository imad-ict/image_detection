from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np


class_labels = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Boat',
    'Truck'
]

f = Path('model_train.json')
model_train = f.read_text()

model = model_from_json(model_train)

model.load_weights('model_weights.h5')
img = image.load_img('', target_size=(32,32))

image_test = image.img_to_array(img)
list_images = np.expand_dims(image_test,axis=0)
results = model.predict(list_images)

single_result = results[0]

most_likely_class_index = int(np.argmax(single_result))
class_likelyhood = single_result[most_likely_class_index]

class_label = class_labels[most_likely_class_index]

print('{} - Likelyhood:{:2f}'.format(class_label,class_likelyhood))
