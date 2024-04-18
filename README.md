# Path Coverage for ROS2

Path coverage is needed for applications like cleaning or mowing where an environment must be fully covered by a robot.
This ROS package executes a coverage path for a given area. The area to cover is given by a polygon which points are set from RViz with "Publish Point". When a successive point equals the first point the area of the polygon is 
divided into cells by an algorithm which resembles the output of the Boustrophedon Cellular Decomposition, so each cell can be covered with simple back and forth motions. The resulting goal points are then given to the navigation stack:

![path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/path_coverage.gif)

the algorithm has been modified to include modes. depending on your choice. mode 3 does not use cellular decomposition and just generates path on the entire costmap. while mode 2 and mode 1 uses cellular decomposition however, the main difference is that mode 2 uses the declared resolution as step size while mode 1 has scale factor that makes the path bigger or smaller as well as offset which adds a fixed length to the horizontal and vertical lines generated. 


Mode 1:
normal opperation of mode 1 without offset and scale shown in the gif above. however, the offset increases the typical horizontal and vertical length by a fixed amount stated in the launch parameter: 

![mode1 path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/mode1_no_scale_but_offset.png)

if scale and offset is simultaneously applied, the triagle shaped lines will be no more:

![mode1 path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/mode1_scale.png)

Mode 2:

![mode2 path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/mode2.png)

Mode 3:


![mode3 path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/mode3.png)

![mode3 path coverage demonstration](https://github.com/BirfenArge/path_coverage/blob/viro_cov_planner/images/mode3_example2.png)



## UPDATE
* The areas to cover needs to be ordered more intelligently (travelling salesman problem). 
    Solved: Polygon list is re-ordered. polygons that share a border/point or at some thresholded distance away from each other are considered neighbours.
    Further works: other means of calculating distance between polygons can be employed. I have used euclidean distance. others exists. some even specifically for polygons.
* Tiny polygons exist and are removed/deleted: robot plans paths to tiny polygons that barely have two coordinates in them.
    Solution: considered polygon's areas and thresholded/cut-off polygons based on some reasonably determined number. This also shortens overall coverage completion time.
* Local planner tends to replan path between waypoints: the paths being followed by the robot ends up being a non straight path which is not good for a cleaning robot.
    Solution: A flexible number of mid waypoints can now be inserted automatically between distant coverage generated waypoints to make for a better path line.

## Requirements
- ROS2 humble or galactic "tested on humble"
- python-shapely
- python-numpy
- ruby for the Boustrophedon Decomposition: $ sudo apt-get install ruby-full

## Usage
1. Launch path coverage: 
    TERMINAL-ROS2: source /opt/ros/humble/setup.bash; source ros2_ws/install/setup.bash; ros2 launch path_coverage path_coverage.launch.py
2. Open RViz, add a *Marker plugin* and set the topic to "path\_coverage\_marker"
3. On the map in RViz, think of a region that you like the robot to cover
4. Click *Publish Point* at the top of RViz
5. Click a single corner of n corners of the region
6. Repeat step 5 and 6 above for n times. After that you'll see a polygon with n corners.
7. The position of the final point should be close to the first
8. When the closing point is detected the robot starts to cover the area

## ROS Nodes
### path\_coverage\_node.py
The node that executes the Boustrophedon Decomposition, calculates the back and forth motions and writes waypoints to a .yaml file.

#### Input: Subscribed Topics
* "/clicked\_point" - Clicked point from RViz
* "/global\_costmap/costmap" - To detect obstacles in path
* "/local\_costmap/costmap" - To detect obstacles in path

#### Output: 
* "pose_output.yaml" - in the home directory. contains waypoints.

### Parameters
* boustrophedon\_decomposition (bool, default: true)

> Whether to execute the Boustrophedon Cellular Decomposition or just do back and forth motions.

* border\_drive (bool, default: false)

> Whether to drive around the cell first before doing back and forth motions.

* robot\_width (float, default: 0.3)

> Width of each path

* costmap\_max\_non\_lethal (float, default: 70)

> Maximum costmap value to consider free

* base\_frame (string, default: "base\_link")

> The robots base frame

* global\_frame (string, default: "map")

> The robots global frame



## Author
ROS1: Erik Andresen - erik@vontaene.de
ROS2:  Azeez Adebayo - hazeezadebayo@gmail.com



