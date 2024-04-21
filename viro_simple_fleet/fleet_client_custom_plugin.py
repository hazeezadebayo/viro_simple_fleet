
from viro_simple_fleet.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from rclpy.duration import Duration
import time, math, rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, LaserScan

# USAGE:
# Terminal 1:
# cd docker_ws/env/dev1
# export DISPLAY=:0.0
# xhost +local:docker
# docker-compose up --build
# or
# docker run -it dev1_tailscaled /bin/bash

# Terminal 2:
# docker exec -it dev1_tailscaled_1 bash
# cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; export TURTLEBOT3_MODEL=waffle; ros2 launch viro_simple_fleet fleet_client.launch.py

# Terminal 3:
# docker exec -it dev1_tailscaled_1 bash
# cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; python3 viro_simple_fleet/scripts/fleet_mngr_main.py


class FleetClientCustomPlugin():
    """
        FleetClientCustomPlugin
    """
    def __init__(self):
        self.navigator = None
        self.nav_start = None
        self.cancel_goal = False
        self.skip_goal = False
        self.continue_goal = False
        self.io_state = True
        self.interrupt = False
        self.pre_recovery_timeout = 200.0
        self.recovery_retry_counter = 0
        self.recovery_retry_max = 3
        self.goal_angle_tol = 45 # [degrees]
        self.goal_distance_tol = 0.45 # [m]
        self.quat_z, self.quat_w, self.quat_x, self.quat_y = 0.0, 0.0, 0.0, 0.0
        self.pose_x, self.pose_y, self.pose_th = 0.0, 0.0, 0.0

    # ................................
    # ...... HELPER FUNCTIONS ........
    # ................................

    def get_distance_to_goal(self, goal_id):
        """
        Get the distance between the current x,y coordinate and the desired x,y coordinate. The unit is meters.
        """
        distance_to_goal = math.sqrt(math.pow(goal_id[0] - self.pose_x, 2) + math.pow(goal_id[1] - self.pose_y, 2))
        return distance_to_goal

    def get_heading_error(self, goal_id, docked=True):
        """
        Get the heading error in radians
        """
        # assume this is a [dock] action.
        delta_x = goal_id[0] - self.pose_x
        delta_y = goal_id[1] - self.pose_y
        # if docked is true, then we have docked already and...
        # we wanna drive backward i.e. [undock] reverse.
        if docked is True:
            delta_x = self.pose_x - goal_id[0]
            delta_y = self.pose_y - goal_id[1]

        desired_heading = math.atan2(delta_y, delta_x)
        heading_error = desired_heading - self.pose_th
        # Make sure the heading error falls within -PI to PI range
        if heading_error > math.pi:
            heading_error = heading_error - (2 * math.pi)
        if heading_error < -math.pi:
            heading_error = heading_error + (2 * math.pi)
        return heading_error

    def get_radians_to_goal(self, goal_id):
        """
        Get the yaw goal angle error in radians
        """
        yaw_goal_angle_error = goal_id[2] - self.pose_th
        return yaw_goal_angle_error

    def position_cb(self, msg):
        """ track robot pose with odom position callback """
        self.quat_z = msg.pose.pose.orientation.z
        self.quat_w = msg.pose.pose.orientation.w
        self.quat_x = msg.pose.pose.orientation.x
        self.quat_y = msg.pose.pose.orientation.y
        self.pose_th = self.quaternion_to_euler(self.quat_x, self.quat_y, self.quat_z, self.quat_w)[-1]
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    # def euler_from_quaternion(self, x, y, z, w):# Convert a quaternion into euler angles (roll_x, pitch_y, yaw_z) - counterclockwise, in radians.
    #         """ x y z w --> yaw """
    #         t3 = +2.0 * (w * z + x * y)
    #         t4 = +1.0 - 2.0 * (y * y + z * z)
    #         yaw_z = math.atan2(t3, t4)
    #         return yaw_z # in radians

    def quaternion_to_euler(self, x, y, z, w):
        """ x y z w --> roll, pitch, yaw """
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def euler_to_quaternion(self, yaw, pitch, roll):
        """ special: radians (yaw pitch roll) -->  [qx, qy, qz, qw]
            abs because I want to use it to define tolerance hence + and - ignored. the mag will be used.
        """
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw] # print([qx, qy, qz, qw])

    # ................................
    # ................................
    # ................................

    def on_startup(self, init_pos_x_y_th):
        """ publish initial pose and  """
        # initialize node capabilities.
        # self.node_ = rclpy.create_node('fleet_client_plugin')
        self.navigator = BasicNavigator()
        self.nav_start = self.navigator.get_clock().now()
        # Add subscription and publishers to the existing node
        # self.odom_sub = self.node_.create_subscription(Odometry,"odom", self.position_cb, 1)
        # self.amcl_sub = self.node_.create_subscription(PoseWithCovarianceStamped,"/amcl_pose", self.position_cb, 1)
        # self.cmd_vel_pub = self.node_.create_publisher(Twist,'/cmd_vel', 1)
        # self.battery_pub = self.node_.create_publisher(BatteryState, 'battery_status', 10)
        self.navigator.odom_sub = self.navigator.create_subscription(Odometry, "odom", self.position_cb, 1)
        self.navigator.cmd_vel_pub = self.navigator.create_publisher(Twist, '/cmd_vel', 1)
        self.navigator.battery_pub = self.navigator.create_publisher(BatteryState, 'battery_status', 10)
        self.navigator.pose_pub = self.navigator.create_publisher(PoseStamped,'/goal_pose', 1) # single destination
        # self.navigator.init_pose_pub = self.navigator.create_publisher(PoseWithCovarianceStamped,'/initialpose', 1)
        self.navigator.get_logger().info('plugin node initialized') # self.node_.get_logger().info('plugin node created')
        # publish init pose
        q = self.euler_to_quaternion(float(init_pos_x_y_th[2]), 0.0, 0.0)
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = float(init_pos_x_y_th[0])
        initial_pose.pose.position.y = float(init_pos_x_y_th[1])
        initial_pose.pose.orientation.z = q[2]
        initial_pose.pose.orientation.w = q[3]
        self.navigator.setInitialPose(initial_pose)
        # ensure amcl is up and running
        # self.navigator.waitUntilNav2Active()
        time.sleep(3)

    # ................................
    # .......... STOP ................
    # ................................

    def stop(self):
        """ cancel or stop the nav task """
        # check interrupt because if for any reason, negotiate path etc. respawns the main function
        # we dont want multiple while loops running. so we want to hopefully terminate one first
        # before calling another. so update interrupt
        self.interrupt = True
        # reset my timer
        self.nav_start = self.navigator.get_clock().now()
        # cancel or stop the nav task
        self.navigator.cancelTask()

    # ................................
    # ........ CHANGE MAP ............
    # ................................

    def load_map(self, map_yaml):
        """ change the current map e.g. elevator states etc. """
        self.navigator.changeMap(map_yaml)
        self.navigator.clearAllCostmaps()

    # ................................
    # ........ RECOVERY ..............
    # ................................

    def recovery(self):
        """
            i have assumed that X amount of time is good enough to go from point A to point B.
            if X amount of time is exceeded and we are still on the road to point B.
            Then robot is stuck therefore reverse.
        """
        # initialize allowed time
        time_allowed = 5 # [secs]
        # cancel the navigation first
        self.stop()
        # Robot is clearing costmaps.
        self.navigator.clearAllCostmaps()
        # Robot hit a dead end, back it up
        self.navigator.backup(backup_dist=0.5, backup_speed=0.1, time_allowance=time_allowed)
        # sleep for the same amount of time + some tolerance.
        time.sleep(time_allowed+1)
        # handle success or failure
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return True
        else:
            return False

    # ................................
    # ........... DRIVE ..............
    # ................................

    def drive(self, goal_x_y_z_w):
        """
            Go to our destination/goal pose
            return success or failure
            if failure, increment number of retries
        """
        # when did we start navigating?
        self.nav_start = self.navigator.get_clock().now()
        # convert goal to posestamped message for nav2
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.nav_start.to_msg()
        goal_pose.pose.position.x = float(goal_x_y_z_w[0])
        goal_pose.pose.position.y = float(goal_x_y_z_w[1])
        goal_pose.pose.orientation.z = float(goal_x_y_z_w[2])
        goal_pose.pose.orientation.w = float(goal_x_y_z_w[3])

        # like publish target as single goal.
        # self.navigator.goToPose(goal_pose)
        # self.navigator.followWaypoints([goal_pose])
        # self.navigator.goThroughPoses([goal_pose])
        # while we are enroute
        condition = False
        while not condition:
        # while not self.navigator.isTaskComplete():
            # where is the robot?
            self.navigator.pose_pub.publish(goal_pose)
            rclpy.spin_once(self.navigator, timeout_sec=10)
            # check interrupt:
            if self.interrupt is True:
                # since we have published a goal earlier and the goal is somewhat
                # unrelated from the instance of navigator. then we need to find a
                # way to cancel the goal. so, publish robot's current pose as target.
                goal_pose.pose.position.x = float(self.pose_x)
                goal_pose.pose.position.y = float(self.pose_y)
                goal_pose.pose.orientation.z = float(self.quat_z)
                goal_pose.pose.orientation.w = float(self.quat_w)
                self.navigator.pose_pub.publish(goal_pose)
                # the easiest way, would have been to just use navigator instance like
                # self.navigator.followWaypoints([goal_pose]) then cancel/stop would work.
                # reset interrupt
                self.navigator.get_logger().info("[plugin]:-interrupted-. \n")
                return False
            # because i used followwaypoints, somehow it percists even after
            # essentially, if we have taken too long and still havent reached our goal...
            if (self.navigator.get_clock().now() - self.nav_start) > Duration(seconds=self.pre_recovery_timeout):
                # increment 'self.num_recovery_retry_counter' since we already failed the actual goal.
                self.recovery_retry_counter += 1
                # in case of a problem, we wanna reverse yeah. or we could spin. doesnt matter.
                result = self.recovery()
                if result is False:
                    # we need not try anymore.
                    return False
                else:
                    return True
            # # if robot is within our softer tolerance limit specified below, go to the next goal.
            d = math.sqrt((float(goal_x_y_z_w[0]) - float(self.pose_x))**2 + (float(goal_x_y_z_w[1]) - float(self.pose_y))**2)
            # print("[samwise-gamgee]: ----------------------------------------------- ")
            # print("[samwise-gamgee]: champ!! ---------2: ", d)
            if d < self.goal_distance_tol:
                if (abs(self.quaternion_to_euler(self.quat_x,self.quat_y,self.quat_z,self.quat_w)[-1]) <= \
                    abs(self.quaternion_to_euler(0.0,0.0,goal_x_y_z_w[2],goal_x_y_z_w[3])[-1]) + math.radians(self.goal_angle_tol) and \
                        abs(self.quaternion_to_euler(self.quat_x,self.quat_y,self.quat_z,self.quat_w)[-1]) >= \
                            abs(self.quaternion_to_euler(0.0,0.0,goal_x_y_z_w[2],goal_x_y_z_w[3])[-1]) - math.radians(self.goal_angle_tol)):
                    # please dont forget to reset the retry counter as it signals total task success.
                    self.navigator.get_logger().info("[plugin]:-reached-. \n")
                    self.recovery_retry_counter = 0
                    condition = True
                    return True

        # result = self.navigator.getResult()
        # print("[samwise-gamgee]: champ!! : ", result)
        # if result == TaskResult.SUCCEEDED:
        #     self.navigator.clearAllCostmaps() # also have clearLocalCostmap() and clearGlobalCostmap()
        #     return True

    # ................................
    # ............ DOCK ..............
    # ................................

    def dock(self):
        """
            Usually there would be some detection here.
            so like [lidar] detects a v-shape and then a controller moves the robot to align with that dock station.
            or like [camera] detects an ar-tag and then a controller moves the robot to the tag and then stops
            here is a simple controller that drives the robot 3m to the front.
            Assumming that every dock location is 3m from the landmarks.
        """
        self.navigator.get_logger().info("[plugin]:-dock call recieved-. \n")
        # distance to move forward
        d = 0.5  # meters
        # calculate new position
        goal_th = self.quaternion_to_euler(0.0, 0.0, self.quat_z, self.quat_w)[-1]
        goal_x = self.pose_x + d * math.cos(goal_th)
        goal_y = self.pose_y + d * math.sin(goal_th)
        # orientation remains the same
        goal_idx = [goal_x, goal_y, goal_th]
        # initialize cmd_vel msg:
        cmd_vel_msg = Twist()
        # some function variables:
        heading_tolerance = 0.04
        yaw_goal_tolerance = 0.07
        goal_distance_tol = self.goal_distance_tol
        # initialize some temp variable
        goal_achieved = False
        while not goal_achieved:
            rclpy.spin_once(self.navigator, timeout_sec=10)
            # check interrupt
            if self.interrupt is True:
                # reset interrupt
                return False
            # -------------------------------------
            distance_to_goal = self.get_distance_to_goal(goal_idx)
            heading_error    = self.get_heading_error(goal_idx)
            yaw_goal_error   = self.get_radians_to_goal(goal_idx)
            # -------------------------------------
            # If we are not yet at the position goal
            if (math.fabs(distance_to_goal) > goal_distance_tol):
            # -------------------------------------
            # If the robot's heading is off, fix it
                if (math.fabs(heading_error) > heading_tolerance):
                    if heading_error > 0:
                        cmd_vel_msg.linear.x  = 0.01
                        cmd_vel_msg.angular.z = 0.9
                        # print("forward_left::")
                    else:
                        cmd_vel_msg.linear.x  = 0.01
                        cmd_vel_msg.angular.z = -0.9
                        # print("forward_right::")
                else:
                    cmd_vel_msg.linear.x = 0.35
                    cmd_vel_msg.angular.z = 0.0
                    # print("forward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    cmd_vel_msg.linear.x  = 0.01
                    cmd_vel_msg.angular.z = 0.9
                    # print("::forward_left")
                else:
                    cmd_vel_msg.linear.x  = 0.01
                    cmd_vel_msg.angular.z = -0.9
                    # print("::forward_right")
            # -------------------------------------
            # Goal achieved, go to the next goal
            else:
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.angular.z = 0.0
                self.navigator.cmd_vel_pub.publish(cmd_vel_msg)
                goal_achieved = True
                self.navigator.get_logger().info(" [plugin] ----------dock completed---------. \n")
                return goal_achieved

            cmd_vel_msg.angular.z = cmd_vel_msg.angular.z / 9 # turn speed reduction
            self.navigator.cmd_vel_pub.publish(cmd_vel_msg)  # Publish the velocity message | vx: meters per second | w: radians per second
            # self.navigator.get_logger().info(" [plugin] goal_dist, heading_err, yaw_err : ["+str([distance_to_goal, heading_error, yaw_goal_error])+"]. \n")


    # ................................
    # ............ UNDOCK ............
    # ................................

    def undock(self, reverse_goal_x, reverse_goal_y, reverse_goal_z, reverse_goal_w):
        """ undock robot """
        self.navigator.get_logger().info("[plugin]:-undock called-. \n")
        reverse_goal_th = self.quaternion_to_euler(0.0,0.0,reverse_goal_z,reverse_goal_w)[-1]
        # set the undock or reverse target
        goal_idx = [reverse_goal_x, reverse_goal_y, reverse_goal_th]
        # initialize cmd_vel msg:
        cmd_vel_msg = Twist()
        # some function variables:
        heading_tolerance = 0.04
        yaw_goal_tolerance = 0.07
        goal_distance_tol = self.goal_distance_tol
        # initialize some temp variable
        goal_achieved = False
        while not goal_achieved:
            rclpy.spin_once(self.navigator, timeout_sec=10)
            # check interrupt
            if self.interrupt is True:
                # reset interrupt
                return False
            # -------------------------------------
            distance_to_goal = self.get_distance_to_goal(goal_idx)
            heading_error    = self.get_heading_error(goal_idx, docked=True)
            yaw_goal_error   = self.get_radians_to_goal(goal_idx)
            # -------------------------------------
            # If we are not yet at the position goal
            if (math.fabs(distance_to_goal) > goal_distance_tol):
            # -------------------------------------
            # If the robot's heading is off, fix it
                if (math.fabs(heading_error) > heading_tolerance):
                    if heading_error > 0:
                        cmd_vel_msg.linear.x  = -0.01
                        cmd_vel_msg.angular.z = 0.9
                        # print("backward_right::")
                    else:
                        cmd_vel_msg.linear.x  = -0.01
                        cmd_vel_msg.angular.z = -0.9
                        # print("backward_left::")
                else:
                    cmd_vel_msg.linear.x = -0.35
                    cmd_vel_msg.angular.z = 0.0
                    # print("backward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    cmd_vel_msg.linear.x  = -0.01
                    cmd_vel_msg.angular.z = 0.9
                    # print("::backward_right")
                else:
                    cmd_vel_msg.linear.x  = -0.01
                    cmd_vel_msg.angular.z = -0.9
                    # print("::backward_left")
            # -------------------------------------
            # Goal achieved, go to the next goal
            else:
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.angular.z = 0.0
                self.navigator.cmd_vel_pub.publish(cmd_vel_msg)
                goal_achieved = True
                self.navigator.get_logger().info("[plugin]:-undock completed-. \n")
                return goal_achieved

            cmd_vel_msg.angular.z = cmd_vel_msg.angular.z / 9 # turn speed reduction
            self.navigator.cmd_vel_pub.publish(cmd_vel_msg)  # Publish the velocity message | vx: meters per second | w: radians per second
            # self.navigator.get_logger().info(" [plugin] goal_dist, heading_err, yaw_err : ["+str([distance_to_goal, heading_error, yaw_goal_error])+"]. \n")

    # ................................
    # .......... IO STATE ............
    # ................................

    def state_io(self, type):
        """ state check on success of reaching a destination or checkpoint
            type: dock_required | undock_required | elevator_required | waypoint
        """
        # wait for some stuff to trigger
        # could be a door etc.
        # implement your lidar field messages here or something, anything.
        # whatever it is IO
        # like if type == 'dock_required':
        #       check if lift is up:
        #       if lift is up
        #           return True

        # or like if type == 'elevator_required'
        #       check if door is open
        #       if door is open
        #           return true

        # etc.

        # Please NOTE that the code will not move forward unless this io_state
        # returns true. because it is a while io_state blah blah blah.
        print("[state_io]:::::::::::::::::::")
        time.sleep(0.05)
        # assume it has triggered
        if self.io_state is True:
            # do some stuff
            return True
        elif self.io_state is False:
            # do some more stuff
            # see if those return true
            return False

    # ................................
    # ............ EXIT ..............
    # ................................

    def on_exit(self):
        """ exit or kill plugin node """
        # self.node_.destroy_node()
        if self.navigator is not None:
            self.navigator.destroy_node()
