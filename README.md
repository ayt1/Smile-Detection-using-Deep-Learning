# Smile-Detection-using-Deep-Learning
Training Smile Detection Model using Python in Keras

## Preprocessing Dataset

The original dataset is GENKI4K dataset which can be downloaded from http://mplab.ucsd.edu/wordpress/?page_id=398

This dataset has faces with background, so we need to extract faces first to build a reliable smile detector. To do that, we use extractFaces.py file and save only the faces in a folder. Here, I have used dlib_cnn_face_detector model named as mmod_human_face_detector.dat since it is much more accurate compared to OpenCV's haarcascade face detector and Dlib's default face detector.

Of course not all the faces were detected. Cnn_face_detector was unable to find 6 of the faces and some of the files came out to be null at the output. So I have coded another script called cleanFaces.py to remove null images and create labels corresponding to remaining images. Now we can move to the training part.

## Training Smile Detector

As an architecture, I have used MiniVGGNet.py which inspired by VGGNet architecture. It may take a long time to train the model if you dont have any GPU. You can use any platform which provides you free GPU like Kaggle kernel, Google Collaboratory etc.I have used Google Collaboratory to train and save the model. In my case, I obtained %91 accuracy. 

## Testing Model in Real Time

After training and saving your model, you can test it in real time to see how it performs using smile_detection.py file. 
