import os
import cv2
import argparse
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from resize import Resize	
from imagetoarray import ImagetoArray
from datasetloader import DatasetLoader
from keras.optimizers import SGD
from arch1 import Arch1
from arch2 import Arch2
from arch3 import Arch3
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
