from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc

import cv2 as cv
import numpy as np
import time
import datetime
import imutils
import facenet
import detect_face
import os
import pickle

from imutils.video import VideoStream
from imutils.video import FPS
from IPython.display import clear_output

class Camera():
    ipcam = None

    def initialise(object):
        global ipcam
	print("Starting video stream")
        #ipcam = VideoStream("rtsp://10.10.25.128:554/11").start() #ip address of camera
	ipcam = cv.VideoCapture(0)

    def close(object):
        global ipcam
	print("Stopping")
        ipcam.release()
        cv.destroyAllWindows()

    def read(object):
        global ipcam
        while True:
	    ret, img = ipcam.read()
            img = cv.resize(img, (1024,600), interpolation=cv.INTER_CUBIC) #resizes camera window
	    cv.imshow('Feed',img) #displays window
	    face_detect(img)
            key = cv.waitKey(1)
	    if key == (27): #press esc to close
		webcam.close()
		break

def face_identify():
    global ipcam
    identifiedPerson = ""
    #modeldir = './model/20170511-185253.pb'  #Original model
    modeldir = './model/20180402-114759.pb'   #Facenet model - breaks code
    classifier_filename = './class/classifier.pkl'
    npy='./npy'
    train_img="./train_img"
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 2
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            
            People = os.listdir(train_img)
            People.sort()
	    Person = 0
    
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

	    c = 0
	    searching = 0

	    
	    
            print('Starting Facial Recognition')
            prevTime = 0
            while True:
                ret, img = ipcam.read()
    
                img = cv.resize(img, (0,0), fx=0.75, fy=0.75)    #resize frame (optional)
    
                curTime = time.time()+1    # calc fps
                timeF = frame_interval
    
                if (c % timeF == 0):
                    find_results = []
    
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
		    
                    if nrof_faces > 0:
			print(' ')
			print('_______Identifying_______')
		 	print(' Number of faces detected: %d' % nrof_faces)
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))
    
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                                print('Face is too close. Plz move back or it will crash :(')
                                continue

                            cropped.append(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv.resize(scaled[i], (input_image_size,input_image_size),
                                                   interpolation=cv.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            
				#Predicting the best match
			    predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
			    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
			    
   				
    			    
                            # print(best_class_probabilities)
                            if best_class_probabilities>0.53: #Controls minimum accuracy
				#Sorting top predictions
			        top = np.argsort(-predictions)
    			        top_one = top[0][0]
	    	       	        top_two = top[0][1]
	       		        top_three = top[0][2]
			    
			        #Makes it more accurate for top result
			        #predictions[0][top_one] = (predictions[0][top_one] * 10) * 2			   

			        print ('Top closest matches are:', predictions[0][top_one],',', predictions[0][top_two],',', predictions[0][top_three])
			        print ('Top three people are:', People[top_one],',', People[top_two],',', People[top_three])
			    	identifiedPerson = People[top_one]
			        Person = best_class_indices[0]
                                # print("predictions")
                                print('Looks like,', People[top_one],' with accuracy of ',predictions[0][top_one])                                
				cv.rectangle(img, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
    				
                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                print('Hey ',People[Person])
                                for H_i in People:
                                    if People[best_class_indices[0]] == H_i:
                                        result_names = People[best_class_indices[0]]
                                        cv.putText(img, result_names, (text_x, text_y), cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 255, 0), thickness=2, lineType=2)
			    
			
                    else:
			searching += 1
			if searching == 75:
			    print("Searching...")
			    searching = 0    
                
                cv.imshow('Feed', img)
	        key = cv.waitKey(1)
	        if key == (27): #press esc to close
   	            webcam.close()
	            break


def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def face_detect(img):
    global ipcam
    box_enlarge_x = 300
    box_enlarge_y = 300
    face_cascade = cv.CascadeClassifier('./Cascade/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv.CascadeClassifier('./Cascade/haarcascade_eye.xml')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    size = img.shape
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 2, minNeighbors = 5) #This causes major fps drop when scaleFactor is at 1.2
    for x,y,w,h in faces:
	det = False
        min_x = clamp(x - box_enlarge_x, 0 , size[1])
	max_x = clamp(x + w + box_enlarge_x, 0 , size[1])
	min_y = clamp(y - box_enlarge_y, 0 , size[0])
	max_y = clamp(y + h + box_enlarge_y, 0 , size[0])

	img = cv.rectangle(img, (min_x,min_y), (max_x, max_y), (0,255,0),1)
	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    	for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	    det = True #Checks that eyes are shown in the image
		       #Think there needs to be a check that cv.rectangle is not empty
            cropped_face = img[min_y : max_y , min_x : max_x]
            cv.imshow('Face',cropped_face)

	now = datetime.datetime.now()
	dateString = now.strftime("%Y-%m-%d %H:%M:%S")
	if det == True:
	    cv.imwrite("/home/research/Documents/PaxIdentity_Main/raw_images/%s.jpg" % dateString, cropped_face)
	#Take snaps of faces detected ^^

#Start program
webcam = Camera()
webcam.initialise()

while True:
    try:
        menu = int(raw_input("Press '1' for Face Detection or '2' for Facial Recognition. "))
    except ValueError:
	print("Sorry, please try again. ")
	continue
    if menu > 2:
	print("Sorry, please try again.")
    else:
	break    

if menu == 1:
    print("You chose face detection. ")
    webcam.read()
elif menu == 2:
    print("You chose face identification. ")
    face_identify()
    




