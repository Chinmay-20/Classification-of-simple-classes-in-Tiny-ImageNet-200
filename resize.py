import imutils
import cv2

class Resize:
	def __init__(self,width,height,inter=cv2.INTER_AREA):
		self.width=width
		self.height=height
		self.inter=inter


	def process(self,image):
		(h,w)=image.shape[:2]
		dW=0
		dH=0

		if w<h: 	#determine shortest dimeension 
			image=imutils.resize(image,width=self.width,inter=self.inter)
			'''
			print(image.shape)
			print(image.shape[0])
			print(image.shape[1])
			'''
			dH=int((image.shape[0]-self.height)/2.0)
		else:
			image=imutils.resize(image,height=self.height,inter=self.inter)
			'''
			print(image.shape)
			print(image.shape[0])
			print(image.shape[1])
			'''
			dW=int((image.shape[1]-self.width)/2.0)

		#we have resized image andd now re grab widdth & heeight use delta to crrop center of image

		(h,w)=image.shape[:2]
		#print(dH:h-dH)
		#print(dW:w-dW)

		image=image[dH:h-dH, dW:w-dW]

		return cv2.resize(image,(self.width,self.height),interpolation=self.inter)



