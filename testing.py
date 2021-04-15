#from imagetoarraypreprrocessor import ImagetOArrayPreproessor
#from simplepreprocessor import SimplePreprocessor
#from simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
from keras.preprocessing.image import img_to_array

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image to be predicted")
ap.add_argument("-m","--model",required=True,help="path to load model")
args=vars(ap.parse_args())

data=[]
labels=["bullettrain","dugong","elephant","espresso","lemon","lion","penguin","potterwheel","schoolbus","steelarchbridge","watertower"]
image=cv2.imread(args["image"])
cv2.imshow("zx",image)
cv2.waitKey(0)
image=cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA)


data.append(img_to_array(image,data_format=None))
data=np.array(data)
data=data.astype("float")/255.0

print("[INFO] loading pre-trained network...")		
model=load_model(args["model"])

pred=model.predict(data,batch_size=1)
print(pred)
print(max(max(pred)))


