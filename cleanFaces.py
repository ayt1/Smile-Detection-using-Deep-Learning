#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from imutils import paths
import os


# In[ ]:


facesPath = 'path_to_extracted_faces_file'
cleanedPath = 'path_to_output_file'
labelPath = 'path_to_label.txt_file'


# In[ ]:


impaths = sorted(list(paths.list_images(facesPath)))


# In[ ]:


null = 0
labels = []
smile = True
for impath in impaths:
    im_name = os.path.split(impath)[-1]
    image = cv2.imread(impath)
    
    if image is not None:
        cv2.imwrite(cleanedPath+im_name, image)
        if smile:
            labels.append('1')
        else:
            labels.append('0')
            
        if im_name == 'file2162.jpg': # the last smiling face
            smile = False
    else:
        null+=1


# In[ ]:


print('Number of null images: ', null)


# In[ ]:


with open(labelPath, 'w') as f:
    for label in labels:
        f.write(label)
        f.write('\n')
f.close()        

