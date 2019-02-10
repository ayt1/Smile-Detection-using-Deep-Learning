#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import dlib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import timeit


# In[ ]:


smile_detector = load_model('/home/itouch/Downloads/smile_detector_MVGGNet.hdf5')
face_detector = dlib.get_frontal_face_detector()


# In[ ]:


cap = cv2.VideoCapture(0)


# In[ ]:


while cap.isOpened():
    ret, frame = cap.read()
    if frame is not None:        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        
        if len(faces) != 0:
            for d in faces:               
                face = gray[d.top():d.bottom(), d.left():d.right()]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0 # scale to range [0,1]
                face = img_to_array(face) # convert to the model compatible format
                face = np.expand_dims(face, axis=0)
                
                tic = timeit.default_timer()
                (notSmiling, smiling) = smile_detector.predict(face)[0]
                toc = timeit.default_timer()
                print('Time to detect smile: {:.2f}'.format(toc-tic))
                label = "Smile" if smiling > notSmiling else "No_Smile"
                
                cv2.putText(frame, label, (d.left(), d.top() - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()),(0, 0, 255), 2)
        else:
            print('No face found !')
            
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:   # exit when ESC is pressed
            break
    
    else: # frame is null
        print('unable to read next frame')
        break

cap.release()
cv2.destroyAllWindows()

