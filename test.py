from keras.models import model_from_json
import os,numpy
from keras.preprocessing import image

def classify(img,file):

    img= image .load_img(img,target_size=(64,64))
    img = image.img_to_array(img)
    img =  numpy.expand_dims(img,axis=0)
    result = model.predict(img)
    if result[0][0] == 1:
        print("Nami", file)
    else:
        print("Luffy",file)




model=open(r"D:\Pantech ai\basic pgm in keras & tensorflow\image classification\model.json",'r')
model = model.read()
model = model_from_json(model)

model.load_weights(r"D:\Pantech ai\basic pgm in keras & tensorflow\image classification\model.h5")

path = r"D:\Pantech ai\basic pgm in keras & tensorflow\image classification\dataset\test"
for dir,sub_dir,files in os.walk(path):
   for file in files:
       classify(os.path.join(path,file),file)
