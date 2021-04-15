# Classification-of-simple-classes-in-Tiny-ImageNet-200

Descritpion : The main idea of project is to understand image classification using deep learning. The project is built in python programming language. The libraries used are tensorflow and keras. A deep learning model is trained on 11 classes (bullet_train, dugong, elephant, espresso, lemon, lion, penguin, schoolbus, steelarchbridge, water tower) of Tiny-ImageNet-200 2015 dataset. 

Directory information:
simple : contains training dataset
images : contains images used for testing the model.(this images are from validation set of Tiny-ImageNet-200)
output2 : contians output of model(graph,weights,model architecture, output[redirected from terminal])

modules information:
arch2.py : code of model
datasetloader, imagetoarray, resize : to load datase
modules : import of all modules required for script.py
script : contains main function 
testing : to load sinngle image and classify image 
log.txt : conts output of timages tested using testing.py

commands:
python script.py --dataset <path_to_dataset>simple --arch <path_to_store-arch_layout>output/arch.png --model<path_to_save_weights>output/weight.hdf5 --graph <path_to_store_graph>output/graph.png 

python testing.py --model<path_to_load_weight>weight.hdf5 --image <path_to_load_image_for_testing>1.JPEG
