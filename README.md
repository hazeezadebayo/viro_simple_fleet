# Viro Simple Fleet Manager (in progress)

This is a ros2 package for AGV fleet management based on Postgresql DB for data storage. The package has two parts, client and master. clients are the robots themselves and the launch file provided is intended for use by each robot to spawn as many robots as possible into the required fleet. However, only one instance of the manager or master may be started. The master provides a CLI for simple understanding and API integration.



## Requirements:
- postgresql db
- Nav2



## Database Table Architecture

The robot table contains the following information.

-`robot_id`: PRIMARY KEY, robot serial number may be used here to uniquely identify the client.

-`fleet_id`: this parameter defines which checkpoints or docks are made available to this client.

-`current_pose`: holds the robot's actual pose on the map in the format pose_x, pose_y and pose_th at any time.

-`agv_status`: holds the current floor identity as well as the client's status 'active/inactive/idle'. which is like assigned a task and active, assigned a task but currently paused, and lastly free to undertake a task respectively.

-`checkpoints`: holds the path identity generated by the master/manager to be traversed by the client e.g. A1, A2, A3. also known as `base`.

-`agv_itinerary`: holds the corresponding actual location of the path declared to be traversed in checkpoints as in '[x,y,z,w],[x,y,z,w],[x,y,z,w]'.

-`landmark`: holds task-specific details like
    - `type`: ['transport', 'loop', 'move', 'charge', 'clean'],
    - `priority`: ['low', 'high', 'medium'], and
    - `landmark`: [specific checkpoint identities that define the task as in 'charge_dock', 'station_dock', 'home_dock'].

-`traffic`: holds reserved locations for which a client is currently entitled. if A1 is reserved by client 1, then if client 2 wants to use checkpoint A1 he would have to wait. Or they will negotiate a path if a robot stand-off is a possibility.

-`last_checkpoint`: holds paths untraversed but will be traversed in the future. essentially a mirror of 'checkpoints' but contains only paths left to be visited also known as `horizon`.

-`m_controls`: holds 'no, False, False, False, False, False' default values, corresponding to whether or not a manual drive is desired [yes or no], and the actual drive command itself [move_left, move_forward, move_right, stop, move_backward] respectively.

-`notifications`: internal and external contents of the simple fleet operation are held here such as other state-related messages:
```bash
    # ------------------------------------------------------------------------
    # dock_required | dock_completed | undock_required | motion
    # no_motion | negotiation_required | negotiation_completed
    # warning_stop | warning_slow | no_warning | undock_completed
    # ------------------------------------------------------------------------
    # elevator_required | startup_failed | startup_success | recovery_required
    # ------------------------------------------------------------------------
```
    the second row contains messages we wish to notify the user via text using Twilio on robot status if Twilio auth_key is properly set up.

-`battery_status`: holds only the battery percent for now.

-`base_map_data`: bytea data type holding the .pgm image of the map base floor

-`floor1_map_data`: bytea data type holding the .pgm image of the map first floor for which an elevator may be used. should ideally only be called in a spot where the location in the base map corresponds to the same robot pose on the first-floor map. 'base-floor:x1,y1,th1 = first-floor:x2,y2,th2'. Where x1, y1, and th1 are elevator request point/position.

-`model_config`: holds client specific parameter wheel_seperation and wheel_radius.

-`shutdown`: 'no' to turn on the system. 'yes' to turn off the system.

-`created_at`: -TIMESTAMP- time of update of any entry on the DB.



## USAGE:

launch turtlebot simulation with:
### Terminal 1:
```bash
$ . /usr/share/gazebo/setup.sh && source install/setup.bash && export TURTLEBOT3_MODEL=waffle && ros2 launch turtlebot3 sim.launch.py
```

### Terminal 2:
launch the viro simple fleet client with the below command after you have filled in the necessary information of the `postgresql` db address and other client-specific information in its launch parameter:

```bash
'hostname': "xxx.xxx.xx.xxx"
'database': "postgres"
'username': "postgres"
'pwd': "xxx"
'port': 123
'robot_id': X
# etc...
```

```bash
$ cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; ros2 launch viro_simple_fleet fleet_client.launch.py
```

### Terminal 3:
similarly, fill the master/server/manager messaging `twilio` information as well as the same `postgresql` db address, before you run the user command line interface with the below command and follow the prompt. A sample graph; with respective nodes and their connecting edges is auto generated when you use the `fm_add_landmark_request` option to build a graph. the notation `A1`, `A2`, `A3`... are used in identifying 'charge_dock', 'station_dock', 'home_dock', 'waypoint' landmarks and `E1`, `E2` is the expected convention to identify 'elevator_dock'.

```bash
$ cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; python3 viro_simple_fleet/scripts/fleet_mngr_main.py
```

![CLI1](https://github.com/hazeezadebayo/viro_simple_fleet/blob/main/media/b.png)
![CLI2](https://github.com/hazeezadebayo/viro_simple_fleet/blob/main/media/c.png)

![running](https://github.com/hazeezadebayo/viro_simple_fleet/blob/main/media/a.png)
