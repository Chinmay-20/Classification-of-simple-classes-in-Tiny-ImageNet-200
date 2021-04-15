from keras.preprocessing.image import img_to_array

class ImagetoArray:
	def __init__(self,dataFormat=None):
		self.dataFormat=dataFormat
		
	def process(self,image):
		return img_to_array(image,data_format=self.dataFormat)
