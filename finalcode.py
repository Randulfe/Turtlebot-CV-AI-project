#!/usr/bin/env python

'''
Copyright (c) 2015, Mark Silliman
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# TurtleBot must have minimal.launch & amcl_demo.launch
# running prior to starting this script
# For simulation: launch gazebo world & amcl_demo prior to run this script

from __future__ import division
import rospy
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion
import random
import cv2
import sys
import roslib
import time
import tf
import ar_track_alvar
from ar_track_alvar_msgs.msg import AlvarMarkers

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def drawMatches(orbimg1, kp1, orbimg2, kp2, matches):
    """
	Implementation of cv2., credit rayryeng

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = orbimg1.shape[0]
    cols1 = orbimg1.shape[1]
    rows2 = orbimg2.shape[0]
    cols2 = orbimg2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([orbimg1, orbimg1, orbimg1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([orbimg2, orbimg2, orbimg2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        orbimg1_idx = mat.queryIdx
        orbimg2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[orbimg1_idx].pt
        (x2,y2) = kp2[orbimg2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
  
#draw matches implementation ends

'''
This function Inputs the test image directory, as well as an index (for file naming) 
It saves the modified file and outputs the file path
'''
def matchImage(inputImageDir, imgNumber):

	#database of templates for testing
	templateDir = ['/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/mustard.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/peacock.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/plum.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/scarlet.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/rope.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/revolver.png',
			'/home/csunix/el15mggr/catkin_ws/src/lab5/src/Images/wrench.png']
	'''
	#database of templates for turtlebot				
	templateDir = ['/home/turtlebot/catkin_ws/src/project/src/Images/mustard.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/peacock.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/plum.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/scarlet.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/rope.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/revolver.png',
			'/home/turtlebot/catkin_ws/src/project/src/Images/wrench.png']
	'''

	#database of templates for testing
	inputImage = inputImageDir

	'''
	#database of templates for turtlebot
	inputImage = 'inputImage'
	'''

	#Template matching starts and draw round image

	img=cv2.imread(inputImage,1)		#read the input image
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #conver the input image to grayscale

	matching_templatePerformance=[]			#declare an array to store the values associated with the performance of each template compared to the input image

	img_thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)	#use adaptive thesholding to create a binary image

	kernel = np.ones((5,5),np.float32)/25		#create kernel for filter

	img_blur = cv2.filter2D(img_thresh,-1,kernel)	#reduce noise by bluring image using filter

	edges = cv2.Canny(img_blur,200, 400)			#use canny edge detection on image

	contour = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]	#returns the contours of the edge detection

	i=0			#loop counter
	bestMatch = 0						#this value is the performance measure of the template that best fits the image
	bestTemplate = 'No match found'		#this value is the file bath of the template that best fits the input image
	bestRectX=0							#declares the x coordinate of the top left point of the rectangle corresponding to the bounding rectangle of the best match
	bestRectY=0							#declares the y coordinate of the top left point of the rectangle corresponding to the bounding rectangle of the best match
	bestRectW=0							#the width of the rectangle bounding the best match
	bestRectH=0							#the height of the rectangle bounding the best match for one of the templates
	bestMatch_flag=0					#flag that indicates a the best match so far has been found
	
	
	#loops through the 10 contours with the largest bounding rectangle, to find the template within the input image

	while (i<10):

		largestElementArea = 0		#value equalts the current largest bounding rect area in the contour list
		contourIndex=0				#indicates the current contour
		matching_templatePerformanceTemp=[]	#temp list storing the current preformances for each template at a specific contour

		for contourElement in contour:	#loop through all contours in contour list
			
			#evaluate parameters of contours
			(rectX, rectY), (rectW, rectH), theta = cv2.minAreaRect(contourElement)
			
			#x, y, w, h = cv2.boundingRect(contourElement) 
			#cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (100,100,100), 1)
			
			elementArea = rectW * rectH #calculate the area of the bounding box of current contour
			
			if (elementArea > largestElementArea):	#find the largest contour by contour bounding box area
				largestElementArea = elementArea
				largestElement = contourElement		#the element corresponding to the largest contour
				largestIndex = contourIndex			#the index of the largest contor in the contour list
				W_H_ratio = rectW/rectH				#evaluate the hight width ratio of the largest contour

			contourIndex=contourIndex+1				#index increment

		#print W_H_ratio
		
		'''
		Eliminate noise by only considering contours of a certain bounding box area, in this case 5000 pixels squared
		And only include contours with a reasnable contour ratio, this excludes very thin but long contours, or wide and but short ones
		The reasnable contour ratio is deffined by the ratio of a A4 sheet of paper, as the templates the algorithm will be searching for 
		is an A4 sheet
		'''
		if (largestElementArea>1000 and (0.6 < W_H_ratio <1.5)):	#elimanate small areas and values outside a reasnable ratio
			
			bestMatch_flag=0	#flag handleing
		
			largestRectX, largestRectY, largestRectW, largestRectH = cv2.boundingRect(largestElement)  #get the rectangleparameters for the largest contour

			rectMask = np.zeros(img.shape, np.uint8)				#create a mask
			rectMask = cv2.cvtColor(rectMask, cv2.COLOR_BGR2GRAY)	#convert mask to grayscale
			
			#use the largest contour parameters to define the size and position of the mask, i.e. create a white rectangle in the position of the largest contour
			cv2.rectangle(rectMask, (largestRectX, largestRectY), (largestRectX+largestRectW, largestRectY+largestRectH), (255,255,255),-1)

			#elimanate all other image information, appart form the largest contour
			img1 = cv2.bitwise_and(img, img, mask=rectMask)

			max_val_sum=0
			for tempDir in templateDir: #loop through the template directories

				templateDirectory = tempDir
				template = cv2.imread(templateDirectory,0)	#read the template image
				template = cv2.resize(template, (largestRectW, largestRectH))	#resize the template so it is the same size as the largest contour

				res = cv2.matchTemplate(img_gray,template, 5)	#match the template with the largest input image

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)	#returns values for the highest likelyhood for every template and it's position
				
				top_left = max_loc	#positon of highest likelyhood for eack tamplate 

				bottom_right = (top_left[0] + largestRectW, top_left[1] + largestRectH)	#use contour dimensions to determine the bottom_right position
				#cv2.rectangle(img,top_left, bottom_right, (0, 255, 0), 1)	#draw the position of the template guess
				
				max_val_sum=max_val_sum+max_val		#max val is unused
				
				matching_templatePerformanceTemp.append(max_val)
				
				if (max_val > bestMatch):		#find the highest match value, i.e. the most likely to be the correct template 
					bestMatch_flag=1			#flag that a match has been found
					bestMatch = max_val			
					bestTemplate = tempDir		#the best template
					bestRectX, bestRectY, bestRectW, bestRectH = cv2.boundingRect(largestElement)	#the best bounding rectangle dimension 
					best_max_val_sum = max_val_sum	#unused
		
			if bestMatch_flag==1:	#store all the template values when the best template is found, for processing later
				matching_templatePerformance=list(matching_templatePerformanceTemp)
		
		if bestMatch<0.5:	#unless it is a very good match, delete the contour and look at the next largest bounding box area
			del contour[largestIndex]		
		else:	#if the match is very good ext the loop
			break
		
		i=i+1	#increment iteration counter

	cv2.rectangle(img, (bestRectX, bestRectY), (bestRectX+bestRectW, bestRectY+bestRectH), (0,255,0), 2)		#draw the rectangle corespnding to the best bounding rectangle dimensions

	print ' '
	print 'Best Match with template matching:'
	print bestTemplate							#print the string for the directory of the best template
	print ' '

	print matching_templatePerformance

	#Template matching stops


	# ORB detector start
	orbimg2 = cv2.imread(inputImage,1) # trainImage		#read image
	orbimg2 = cv2.cvtColor(orbimg2, cv2.COLOR_BGR2GRAY) #convert to grascale

	ORB_templatePerformance=[]							#decalare list to store preformance values
	ORB_bestMatch = 10000000							#declare value coresponding to performance value of the best match					
	ORB_bestTemplate = '0'								#declare string coresponding to the filepath of the best match
	j=0													#loop counter
	for tempDir in templateDir:		#loop through template directory

		templateDirectory = tempDir
		orbimg1 = cv2.imread(templateDirectory,1)	##read template file	
		orbimg1 = cv2.cvtColor(orbimg1, cv2.COLOR_BGR2GRAY)	#convert template file to grayscale
		# Initiate ORB detector
		orb = cv2.ORB()			#initialice ORB

		# find the keypoints and descriptors with ORB
		kp1, des1 = orb.detectAndCompute(orbimg1,None)		#detect the key points (features) and compute the relative distances for the
		kp2, des2 = orb.detectAndCompute(orbimg2,None)		#detect the key points and compute the 

		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		#bf = cv2.BFMatcher()
		
		# Match descriptors
		matches = bf.match(des1,des2)			#find matches
		#matches = bf.knnMatch(des1,des2, k=2)

		# Sort them in the order of their distance.	
		
		#sort the matches ddepending on the relative distance between each match, so that the shortes distances are at the start of the matches list
		matches = sorted(matches, key = lambda x:x.distance)
		
		distanceSum = 0
		
		'''
		good=[]

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append(m)
				distanceSum = distanceSum + m.distance
				sumCount = sumCount + 1
			if sumCount == 0:
				sumCount=0.0001
		#print good

		distancesSumWithNoise=0


		for m in matches[:60]:
			distancesSumWithNoise = distancesSumWithNoise+m.distance
			sumCount=sumCount+1

		normalDistance = distancesSumWithNoise/sumCount
		#print normalDistance
		'''
		
		sumCount = 0
		for m in matches[:70]:			#only look at 70 of the best matches
			distanceSum = distanceSum + m.distance		#sum the distance of the 70 best matches
			sumCount=sumCount+1

		distanceAv=distanceSum/sumCount		#find the average distance
		##print distanceAv
		
		if (distanceAv<ORB_bestMatch):
			ORB_bestMatch = distanceAv
			ORB_bestTemplate = tempDir
			
		ORB_templatePerformance.append(distanceAv)

		
	print ORB_templatePerformance

	orbimg3 = cv2.imread(ORB_bestTemplate, 0)

	# Draw first 10 matches.
	orbimg4 = drawMatches(orbimg3,kp1,orbimg2,kp2,matches[:70])

	print 'Best Match with feature matching:'
	print ORB_bestTemplate
	print ' ' 

	cv2.imshow('ORB Preview3', orbimg4)

	#orb detector end

	
	# Matching method performance check certainty value combines both results and outputs the final guess

	finalGuess='0'
	
	if (bestMatch<0.2):		#if template match is very bad go with orb detector
		finalGuess = ORB_bestTemplate		#use the orb detector guess
	else:
		
		#as orb detector is a distance based feature matching algorithm, the lower the disance the better the match, i.e a perfect match would have a distamce sum value of 0
		ORB_templatePerformanceSum=sum(ORB_templatePerformance)	#all the distances for each template
		ORB_tp_normalised = [x / ORB_templatePerformanceSum for x in ORB_templatePerformance]	#normalise all the values
		
		#calculate how close the best guess is from the mean, if it is fruther away from the mean then it is better as it is more certain
		ORB_sel_dev=abs((min(ORB_tp_normalised)-np.mean(ORB_tp_normalised))/np.mean(ORB_tp_normalised))	

		#weight the ORB matrix
		ORB_tp_weighted = [x*ORB_sel_dev for x in ORB_tp_normalised]	#
		
		matching_templatePerformanceSum=sum(matching_templatePerformance)
		matching_tp_normalised = [x / matching_templatePerformanceSum for x in matching_templatePerformance]	#highest is best

		matching_sel_dev=abs((max(matching_tp_normalised)-np.mean(matching_tp_normalised))/np.mean(matching_tp_normalised))
		matching_tp_weighted = [x*matching_sel_dev for x in matching_tp_normalised]
		
		print matching_tp_normalised
		print ORB_tp_normalised

		combine_weighted=[a-b for a,b in zip(matching_tp_weighted, ORB_tp_weighted)]	

		templateIndex = combine_weighted.index(max(combine_weighted))
		
		finalGuess=templateDir[templateIndex]


	print ' ' 
	print ' ' 
	print 'Best template:'
	print finalGuess
	print ' ' 
	
	
	#write to the image and print the guesss onto to image

	#cv2.imshow('Image Preview1', img1)
	cv2.imshow('Image Preview', img)
	#cv2.imshow('Image Preview1', template)
	
	imgNumberString = str(imgNumber)
	
	TMtext='Template match: '
	TMtext+=bestTemplate[49:]
	cv2.putText(img,TMtext,(10,370), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
	
	ORBtext='ORB detector: '
	ORBtext+=ORB_bestTemplate[49:]
	cv2.putText(img,ORBtext,(10,410), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
	
	Answertext='Conclusion: '
	Answertext+=finalGuess[49:]
	cv2.putText(img,Answertext,(10,450), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
	
	#for testing
	cv2.imwrite('/home/csunix/el15mggr/Detections/processed'+imgNumberString+'.png', img)
	
	#for turtlebot
	#cv2.imwrite('/home/csunix/el14rbmc/Detections/processed'+imgNumberString+'.png', img)
	
	return finalGuess
	
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()


	
runFlag=True
	
def photoCallback(data):
	global runFlag
	bridge = CvBridge()	
	try:
		cvimage = bridge.imgmsg_to_cv2(data, "bgr8")

	except CvBridgeError:

			pass

	savedImageFilename='/home/csunix/el15mggr/Detections/capture0.png'
	
	'''
	#for testing
	rospy.Subscriber('camera/rgb/image_raw', Image)
	savedImageFilename = '/home/csunix/el15mggr/Detections/capture'
	savedImageFilename += str(self.photoNumber)
	savedImageFilename +='.png'
	'''
	
	'''
	#for turtlebot
	rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
	sevedImageFilename = '/home/turtlebot/Detections/capture'
	savedImageFilename += str(photoNumber)
	savedImageFilename +='.png'
	'''

	if runFlag:
		cv2.imwrite('/home/csunix/el15mggr/Detections/capture0.png', cvimage)
		runFlag=False
	


	


	
	
class GoToPose():
    def __init__(self):                                                                  #init of the class

        self.goal_sent = False                                                           #sets no goal at the init

	                                                                                     # What to do if shut down (e.g. Ctrl-C or failure)
	rospy.on_shutdown(self.shutdown)
	
                                                                                   # Tell the action client that we want to spin a thread by default
	self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
	rospy.loginfo("Wait for the action server to come up")

	                                                                                      # Allow up to 5 seconds for the action server to come up
	self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):                                                            # Move function

                                                                                          # Send a goal (x,y) pos
        self.goal_sent = True
	goal = MoveBaseGoal()                                                                 # init class movegoal in the object goal
	goal.target_pose.header.frame_id = 'map'                                              # sets the frame the goal is and the robot is moving around
	goal.target_pose.header.stamp = rospy.Time.now()									  # stamp??
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),					  # go to this position
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	                                                                                      # Start moving
        self.move_base.send_goal(goal)

	                                                                                      # Allow TurtleBot up to 60 seconds to complete task
	success = self.move_base.wait_for_result(rospy.Duration(60)) 

        state = self.move_base.get_state()                                                # Gets the position of the robot
        result = False                                                                    # inits result as false

        if success and state == GoalStatus.SUCCEEDED:                                     # if time didnt expire and state is the same as goal state 
            # We made it!
            result = True                                                                 # result true
        else:
            self.move_base.cancel_goal()                                                  # else stop moving

        self.goal_sent = False                                                            # once it finishes with good or bad answers it eliminates the goal
        return result    
        	                                                                              # it returns the result (true/false)

    def shutdown(self):                                                                   # when it is shutdown function
        if self.goal_sent:                                                                # if there was a goal
            self.move_base.cancel_goal()                                                  # it is cancelled
        rospy.loginfo("Stop")
        rospy.sleep(1)


cbflag=False

def callback(data):                               # function that is called everytime the robot sees an AR
	global cbflag 
	if (len(data.markers)!=0):                   # if there is an AR, fire the flag
		cbflag=True
	else :
		cbflag=False


	
def explorer():                                  
	rospy.init_node('nav_test', anonymous=False)                                  #init the code node
	
	
	pub = rospy.Publisher('mobile_base/commands/velocity', Twist,queue_size=10)   #moving the robot publisher
	rospy.Subscriber('ar_pose_marker', AlvarMarkers,callback)                  # this calls callback everytime it sees an AR marker
	
	#rospy.Subscriber('ar_pose_marker', ar_marker_0, callback)
	
	
	navigator = GoToPose()                                                     #objects for each corresponding classes                                               
	desired_velocity = Twist()
	br= tf.TransformBroadcaster()
	listener = tf.TransformListener() 
	lol=0
	rate = rospy.Rate(10) #10hz



		
	photoIndexFlag=1                                                  #initialization of variables
	finish=False                                                      # it will tell when the code has to stop (if true)
	ARflag=False                                                      #
	flag2=False                                                       #
	xpointA=0
	ypointA=0
	robotx=0
	#trans1[0]=0
	#trans1[1]=0
	roboty=0		
	i=0
	x=0
	y=0
	u=0
	v=0	
	xposARone=0	
	xposARtwo=0
	l=0
	print("X coodinate \n")            
	x = input("")# SPECIFY X COORDINATE HERE for going to the centre position
	print("Y coordinate \n")
	y = input("")# SPECIFY Y COORDINATE HERE for going to the centre position
	theta = 0# SPECIFY THETA (ROTATION) HERE
	position = {'x': x, 'y' : y}
	quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
	rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
	success = navigator.goto(position, quaternion)
	


	while not rospy.is_shutdown():
		
		print cbflag
		
		if ((cbflag==False) and l<=480  ): 	#if no AR and l is not 480 yet
			l=l+1
			desired_velocity.angular.z=0.3  #turn robot over his z axis and look for AR
			desired_velocity.linear.x=0
			rospy.loginfo("look for AR")
			pub.publish(desired_velocity)
			rate.sleep()

			#time.sleep(1)
			#l=0	
		else:                                #if AR detected
			l=0
			rospy.loginfo(cbflag)        				
			desired_velocity.angular.z=0  
			desired_velocity.linear.x=0
			rospy.loginfo("found AR")
			pub.publish(desired_velocity)
			try:	
				rospy.loginfo("starting")
				c=(trans0,rot0) = listener.lookupTransform('/map', '/base_link', rospy.Time(0)) #get transformation between robot and map
			
				rospy.loginfo("transformation robot")
				

				robotx= trans0[0]
				roboty=trans0[1]	
				
		
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				pass
			
			if cbflag==True: #there is an AR
				
				rospy.loginfo("empezanado")
				try:
					time.sleep(5)
					b=(trans,rot) = listener.lookupTransform('/map', '/ar_marker_0', rospy.Time(0)) #get transformation between AR and map
					
					time.sleep(5)
					rospy.loginfo("initialising...")
										
					br.sendTransform((0.0,0.0,0.5),(0.0,np.sin(180/2.0),0.0,np.cos(180/2.0)),rospy.Time.now(),"pointA","ar_marker_0") #create a frame point A 0.5 meters far away from marker_0 and 90º turned along y axis
					
					rospy.loginfo("...")
					time.sleep(5)
					o=(trans1,rot1)=listener.lookupTransform('/map', '/pointA', rospy.Time(0)) # transform point A relative to the map
					xpointA=trans1[0]
					ypointA=trans1[1]					
					rospy.loginfo(xpointA)
					rospy.loginfo(ypointA)	
					if xposARone==0 :                                                           #if it is the first time it detects an AR
						flag2=True 
						ARflag=True
						xposARone=trans[0]                                                      #set image 1 position and save it
						yposARone=trans[1]
						stringOne = 'X position first Image:\n'
						stringOne +=str(xposARone)
						stringOne +='\nY position first Image:\n'
						stringOne +=str(yposARone)
						file2write=open("/home/csunix/el15mggr/Detections/project.txt",'w')
						file2write.write(stringOne)
						file2write.close()
						rospy.loginfo("AR one")
						
					elif ((trans[0]>= (xposARone-0.5)) and (trans[0]<=(xposARone+0.5))): #if there is already an AR and the one detected is really similar to it
						ARflag=True
						flag2=False
						rospy.loginfo("AR one repeated")                                  #ignore this AR flag
						
					else:                                                                #otherwise is going to be the second AR, do the same as with the first one although this time finish is fired
						finish=True
						ARflag=True
						flag2=True
						xposARtwo=trans[0]
						yposARtwo=trans[1]
						stringTwo = stringOne
						stringTwo +='\n\n\nX position second Image:\n'
						stringTwo +=str(xposARtwo)
						stringTwo +='\nY position second Image:\n'
						stringTwo +=str(yposARtwo)
						file2write=open("/home/csunix/el15mggr/Detections/project.txt",'w')
						file2write.write(stringTwo)
						file2write.close()
						rospy.loginfo("AR one")			
									
				except:
					print 'tranform failed'
					pass
				rospy.loginfo("obtaining points")	
			
			else:
				rospy.loginfo("no AR")	
		
			if cbflag==True and flag2==True:                     #if it is the first time it sees that AR
				rospy.loginfo("move to Point A")                #move towards it
				x=xpointA
				y=ypointA
				
			else:                                              #if it is an already detected AR or there is no AR
				rospy.loginfo("move to random")
				l=0
				i=i+1

				if i<=10:
					u=robotx+1                                #go to a random point relative to robot position in the map
					v=roboty+1
					x=random.uniform(-u, u)
					y=random.uniform(-v, v) 
				
				elif i>10 and i<=15:					
					u=robotx+3
					v= roboty+3
					x=random.uniform(-u, u)
					y=random.uniform(-v, v) 
										
				else:					
					u=robotx+5
					v= roboty+5
					x=random.uniform(-u, u)
					y=random.uniform(-v, v) 

			theta =10                                                  #move the robot                   
			position = {'x': x, 'y' : y}
			quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
			rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
			success = navigator.goto(position, quaternion)

			if success:                                       #if the movent succeed
				ARflag=False
				rospy.loginfo("Completed")
				
				if xpointA==x:                                #if the movement succeed and it went towards a image
					rospy.loginfo("AR found")
					#photo

					time.sleep(5)
					
					rospy.Subscriber('camera/rgb/image_raw', Image, photoCallback) #do image recognision block of code

					print 'sucess save image'
					
					print 'photo finished'
					time.sleep(3)
					filename='/home/csunix/el15mggr/Detections/capture0.png'
					matchImage(filename, photoIndexFlag)
					time.sleep(1)
					photoIndexFlag=photoIndexFlag+1
					if finish:                                    #if last AR taken, get out of the loop and stop running
						return
					#photo end
				else:				
					rospy.loginfo("Random")
					
			else:
				rospy.loginfo("out of the map")
				
		global runFlag
		runFlag=True
				
	rate.sleep()
	
	
	

if __name__ == '__main__':
	try:
		explorer()
	except rospy.ROSInterruptException:
		rospy.loginfo("Ctrl-C caught. Quitting")



