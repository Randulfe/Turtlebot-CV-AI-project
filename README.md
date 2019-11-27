# Computer Vision + Navigation uni project to control Turtle Bots

The pupose of this uni project was to develop a robotics system using ROS (Robotic Operating System) to control a [Turtlebot 2 Robot](https://www.turtlebot.com/). This project was tested on actual robots as part of the assignment process and evaluated in terms of actual robotic performance. 

## The objective
The main specifications were the following: 
⋅⋅* The turtlebot would start in a random position inside a maze.
⋅⋅* The 2D map of the maze will be provided.
⋅⋅* The robot will then have to explore the space looking for pictures left randomly with an AR tag in the most efficient way that it can explore the given map. 
⋅⋅* These pictures will have either a cluedo character or a weapon from the famous game. 
⋅⋅* One the robot finds the picture, it must recenter its position so that the camera can take a snapshot of it and using Computer Vision (CV) determine which character or weapon it is using feature matching and other CV algorithms (in our case we used feature and template matching). 
⋅⋅* It must repeat the process and get all the images left in the maze (four in total). 

## Our solution
The method we used for the navigation was to move the robot within random relative points inside the map. Once it reaches the point it will turn around mapping its surrounding looking for the AR mark with its camera. (We later found out it was not definitely the most efficient navigation method but it worked well enough for a small maze).

Once we found the AR marker the robot would reposition itself so that the camera faces perpendicular to the plane of the picture using the transformation frames of the robot, camera and space. Furthermore, it would store the relative position of that marker so that next time it found it would already know that the picture was already explored.

For the CV part, after taking a snapshot we run template and feature matching and then combined the punctuation from both to ensure the best match. Most of the times it worked although rarely would have some issues with very similar characters. 

Once the CV part had run, we relaunched the navigation algorithm and the process was repeated. 

## Our outcome
We did pretty well in this assignment. We managed to find 3/4 pictures within the time limit given. The only problem we faced was that the navigation algortithm got bugged and the end turning around continuosly not sure whether it remembered the position of one of the previous AR markers in the map.
