import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imageai.Detection import ObjectDetection
import cv2
import numpy as np
import statistics
import pyttsx3


# Global Variables
detector = 0
execution_path = 0



def objdtn_setup(): # Setup the models for object detection (you can choose between three models)
	global execution_path
	global detector
	os.chdir('..')
	execution_path = os.getcwd()
	detector = ObjectDetection()
	# '''
	detector.setModelTypeAsYOLOv3() # For YOLOv3
	detector.setModelPath( os.path.join(execution_path, "Models/yolo.h5"))
	# '''
	'''
	detector.setModelTypeAsRetinaNet() # For RetinaNet
	detector.setModelPath( os.path.join(execution_path, "Models/resnet50_coco_best_v2.0.1.h5"))
	'''
	'''
	detector.setModelTypeAsTinyYOLOv3() # For TinyYOLOv3
	detector.setModelPath( os.path.join(execution_path, "Models/yolo-tiny.h5"))
	'''
	detector.loadModel()




def objdtn(frame_name): # detect objects in the left frame	
	global execution_path
	global detector
	detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'StereoImages/'+str(frame_name)), minimum_percentage_probability=70)
	# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'StereoImages/'+str(frame_name)), output_image_path=os.path.join(execution_path , "Examples_of_Object_Recognition_and_Feature_Matching/ObjectRecognition.png"), minimum_percentage_probability=70) #
	if(detections!=[]):
		prime_object = {}
		max_area = 0
		for eachObject in detections:
			arr_cd = eachObject["box_points"]
			area = (arr_cd[2]-arr_cd[0]) * (arr_cd[3]-arr_cd[1])
			if(area>=max_area):
				max_area = area
				prime_object = eachObject
		return True, prime_object
	else:
		return False, {}





def is_it_in(box_pts, x, y): # function to find if a given point is inside a box or not
	box_pts = list(map(float, box_pts))
	x = float(x)
	y = float(y)
	if(x>=box_pts[0] and x<=box_pts[2] and y>=box_pts[1] and y<=box_pts[3]):
		return True
	else:
		return False




def feature_detector(frame1, frame2, box_pts): # Match features in both the stereo images using SIFT
	global execution_path
	left_im = cv2.imread(os.path.join(execution_path, "StereoImages/"+str(frame1)))
	right_im = cv2.imread(os.path.join(execution_path, "StereoImages/"+str(frame2)))
	sift = cv2.xfeatures2d.SIFT_create()
	kp_1, desc_1 = sift.detectAndCompute(left_im, None)
	kp_2, desc_2 = sift.detectAndCompute(right_im, None)
	index_params = dict(algorithm=0, trees=5)
	search_params = dict()
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(desc_1, desc_2, k=2)
	good_points = []
	# print('The corresponding coordinates: ') #
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			if(is_it_in(box_pts, kp_1[m.queryIdx].pt[0], kp_1[m.queryIdx].pt[1])):
				good_points.append(m)
				# print(str(kp_1[m.queryIdx].pt) + " : " + str(kp_2[m.trainIdx].pt)) #
	# result = cv2.drawMatches(left_im, kp_1, right_im, kp_2, good_points, None) #
	# cv2.imwrite(os.path.join(execution_path, "Examples_of_Object_Recognition_and_Feature_Matching/FeatureMatching.png"), result) #
	ret = [(kp_1[m.queryIdx].pt[0] - kp_2[m.trainIdx].pt[0]) for m in good_points]
	return ret





def main_code(left_frame, right_frame, fd, enj): # Combining all
	truth, prime_object = objdtn(left_frame)
	if(truth==False):
		print("Nothing in frame")
	else:
		diff_arr = feature_detector(left_frame, right_frame, prime_object["box_points"])
		if(diff_arr==[]):
			print("No significant features detected")
		else:
			# print('The array of pixel-difference of x coordinates: ') #
			# print(diff_arr) #
			# print('The bounding box coordinates: ') #
			# print(prime_object["box_points"]) #
			med = statistics.median(diff_arr)
			# print('The median value of pixel-difference: ' + str(med)) #
			distance = fd/med
			print("\n\nFINAL VERDICT:")
			print("There is a " + str(prime_object["name"]) + " at a distance of " + "{:.2f}".format(round(distance, 2)) + "m. \n")
			enj.setProperty('rate', 150)
			enj.say("\nThere is a " + str(prime_object["name"]) + " at a distance of " + "{:.2f}".format(round(distance, 2)) + " meters. \n")
			enj.runAndWait()
			enj.stop()





## Main part of the code: 

engine = pyttsx3.init()
objdtn_setup()


left_frame = 'left1.jpg'
right_frame = 'right1.jpg'


# fd, the camera parameter = focus(same for both the cams) x distance between the cameras
# can be calculated by a pair of stereo images of an object placed at a known distance (say 10m)
# f*d = Distance * (x1 - x2). Let's assume the median pixel difference of the corresponding feature points of the 
# same object in left and right images are 15 pixels.
# so, fd = 10*15 = 150 m-pixel
fd = 321.13 # the value is 321.13 for the set of stereo images I am using, a new stereo camera may have a different fd.


main_code(left_frame, right_frame, fd, engine)
