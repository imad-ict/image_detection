import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

model = vgg16.VGG16()

img = image.load_img('aero.jpg', target_size=(224,224,3))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = vgg16.preprocess_input(x)

predict = model.predict(x)

predited_classes = vgg16.decode_predictions(predict, top=4)

print('Predictions for this image')

for imagenet_id,name,likelihood in predited_classes[0]:
    print('Predictions: {} - {:2f}'.format(name,likelihood))