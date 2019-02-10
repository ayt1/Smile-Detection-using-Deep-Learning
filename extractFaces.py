#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
from imutils import paths
import dlib
import timeit


# In[ ]:


datasetPath = 'path_to_dataset'
outputPath = 'path_to_ouputFile'


# In[ ]:


impaths = sorted(list(paths.list_images(datasetPath)))


# In[ ]:


cnn_face_detector = dlib.cnn_face_detection_model_v1('path_to_mmod_human_face_detector.dat')
second = []


# In[ ]:


fail = 0
for impath in impaths:
    im_name = os.path.split(impath)[-1]
    image = cv2.imread(impath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    tic = timeit.default_timer()
    dets = cnn_face_detector(gray, 1)    
    toc = timeit.default_timer()
    
    second.append(toc-tic)
    
    if len(dets) > 0:        
        for d in dets:
            face = image[d.rect.top():d.rect.bottom(),d.rect.left():d.rect.right()]            
            cv2.imwrite(outputPath+im_name, face)
    else:
        print('No face detected in image: '+im_name)
        fail+=1


# In[ ]:


print('Average time for single detection: ', sum(second)/len(second))
print('Number of the faces not detected: ', fail)


# In[ ]:




