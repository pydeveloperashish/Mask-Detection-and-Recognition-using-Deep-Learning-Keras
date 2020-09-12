#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


prototxtPath=os.path.sep.join([r'C:\Python37\Projects\face-mask-detector\face_detector','deploy.prototxt'])
weightsPath=os.path.sep.join([r'C:\Python37\Projects\face-mask-detector\face_detector','res10_300x300_ssd_iter_140000.caffemodel'])



net=cv2.dnn.readNet(prototxtPath,weightsPath)


model=load_model(r'C:\Python37\Projects\face-mask-detector\mobilenet_v2.model')

image=cv2.imread(r'C:\Python37\Projects\face-mask-detector\examples\example_03.png')


(h,w)=image.shape[:2]


blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))



net.setInput(blob)
detections=net.forward()


#loop over the detections
for i in range(0,detections.shape[2]):
    confidence=detections[0,0,i,2]
    
    
    if confidence>0.5:
        #we need the X,Y coordinates
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype('int')
        
        #ensure the bounding boxes fall within the dimensions of the frame
        (startX,startY)=(max(0,startX),max(0,startY))
        (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
        
        #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
        face=image[startY:endY, startX:endX]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(224,224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face,axis=0)
        
        (mask,withoutMask)=model.predict(face)[0]
        
        #determine the class label and color we will use to draw the bounding box and text
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        #include the probability in the label
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
        #display the label and bounding boxes
        cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(image,(startX,startY),(endX,endY),color,2)
        
        
        
cv2.imshow("OutPut",image)
cv2.waitKey(0)
cv2.destroyAllWindows()



