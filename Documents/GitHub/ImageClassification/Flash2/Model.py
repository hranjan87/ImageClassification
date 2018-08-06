from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import model_from_json
import numpy as np

model = ResNet50(weights='imagenet')
print(model.summary())
img_path="image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)

model_json=model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print('Predicted:', decode_predictions(preds, top=3)[0])