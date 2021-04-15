from modules import *
#train size =4400
#test size= =1100


ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("-a","--arch",required=True)
ap.add_argument("-w","--weight",required=True)
ap.add_argument("-g","--graph",required=True)
args=vars(ap.parse_args())


imagePaths=list(paths.list_images(args["dataset"]))

robj=Resize(32,32)
ia=ImagetoArray()

dl=DatasetLoader([robj,ia])
(images,labels)=dl.load(imagePaths)

classname=np.unique(labels)
classname=list(classname)

images=images.astype("float")/255.0

trainX,testX,trainY,testY=train_test_split(images,labels,test_size=0.2)

'''
for i in trainY:
	print("trainY before ",i)
'''
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)
'''
for i in trainY:
	print("trainY after ",i)
'''

aug=ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


model=Arch2.build(width=32,height=32,depth=3,classes=len(classname))

plot_model(model,to_file=args["arch"],show_shapes=True)
#opt=Adam(lr=1e-3)
model.compile(loss="categorical_crossentropy",optimizer="adam" ,metrics=["accuracy"])

callback=ModelCheckpoint(args["weight"],monitor="val_loss",save_best_only=True,verbose=1)
callback=[callback]

H=model.fit(aug.flow(trainX,trainY,batch_size=32), validation_data=(testX,testY), callbacks=callback, verbose=2,batch_size=32,epochs=75,steps_per_epoch=len(trainX)/32)
#each training sample is augmented only one time and therefore 1000 transformed images will be generated in each epoch.
#the total number of unique images increases in the whole training from start to finish, and not per epoch
#It will always generate new images, no matter how many epochs you have.
#i would have used floww from directory if i had seprate directory for train and test
#in flow batch_size=32 ie one batcchh will have 32 imagesstacked together inn tensor shape of tensor will be (batch_size,width,colu,depth)
#steps_per_epoch=It will always generate new images, no matter how many epochs you have.
pred=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1),target_names=classname))
#print("Accuracy of model is :",accuracy_score(testY,pred))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,75), H.history["loss"],label="training_loss")
plt.plot(np.arange(0,75), H.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0,75), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0,75), H.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Validation acuuraacy/loss")
plt.xlabel("Epoch")
plt.ylabel("Accuraacy/Loss")
plt.legend()
plt.savefig(args["graph"])

#python script.py --dataset simple --architecture output1/arch.png --weight output1/weights.hdf5 --graph output1/graph.png

