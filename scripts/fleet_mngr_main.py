#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
import psycopg2, psycopg2.extras, collections
from matplotlib import cm
from PIL import Image
import time, io, cv2, yaml, os, signal, math, json, heapq, re
from twilio.rest import Client
from pathlib import Path
import matplotlib.pyplot as plt
from threading import Timer

'''
Assumptions:
1. checkpoints are linked by a straight line
2. checkpoints are at least 3.0m away from one another. i.e. at least the length of agv + some_tolerance.
3. elevator -- naming convention -- E1, E2.. represent elevator so if E is in the alphabet -- we request map change.
'''

########################################################################
# Vehicle parameters
LENGTH = 1.0# 4.5  # [m]
WIDTH = 0.5 #2.0  # [m]
BACKTOWHEEL = 0.12 #1.0  # [m] increasing shifts chasis forward.
WHEEL_LEN = 0.1 #0.3  # [m] # single tyre length
WHEEL_WIDTH = 0.05 #0.2  # [m] # single tyre width
TREAD = 0.18 #0.7  # [m] # left to right tyre distance
WB = 0.75 #2.5  # [m] front tyre to back tyre distance
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
robot_id, fleet_id = 0, 'unknown'
checkpoints, agv_itinerary, landmark, graph = [], [], [], {}
task_cleared, task_dictionary = False, {}
agv_config_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
file_path = Path(agv_config_dir+'/viro_simple_fleet/config/fleet_mngr.yaml')
########################################################################


class Ui_MainWindow():
    def __init__(self, config_path=file_path):

        # wether or not to show matplotlib car
        self.show_robot_animation = True
        self.single_robot_view = False
        # Turn on interactive mode
        plt.ion()
        # Create a new figure
        self.fig, self.ax = plt.subplots()
        # initialize helper variables
        self.o_q = []
        self.control_command = None
        self.feedback_current_pose = None
        self.feedback_notifications = None
        self.feedback_last_checkpoint = None
        self.feedback_checkpoints = None
        self.feedback_agv_status = None
        self.feedback_shutdown = None
        self.feedback_controls = None
        self.feedback_model_config = None
        self.feedback_landmarks = None
        self.feedback_battery_status = None
        self.feedback_base_map_img = None
        self.feedback_floor1_map_img = None
        self.nav_map_obtained = False
        # initialize fleet dropdown menu
        self.available_robots = []
        self.decision = ['yes', 'no'] # terminal interaction
        self.job_types =['transport', 'loop', 'move', 'charge', 'clean']
        self.job_priorites = ['low', 'high', 'medium']
        self.station_type = ['charge_dock', 'station_dock', 'home_dock', 'elevator_dock', 'waypoint']
        # initialize co-ordinates! this is for plotting on matplotlib
        self.agv_track_dict = {"x":[], "y":[],}
        # initialize job ids dropdown menu
        self.job_ids = []
        self.landmark_dict = {"x":[], "y":[], "t":[], "k":[],}
        # initilaize path to configuration
        self.file_path = config_path
        self.load_from_yaml()
        # establish connection to DB
        self.conn = psycopg2.connect(host=hostname, dbname=database, user=username, password=pwd, port=port_id)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # initialize
        global graph
        for node in task_dictionary['graph']:
                neighbors = set(tuple(x) for x in task_dictionary['graph'][node])
                graph[node] = neighbors

        self.interactive_robot_fleet_startup()

##################################################
# ---------------------------------------------- #
#      ON START-UP: LOAD FLEET LIST AND MENU     #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- update_current_agv  ---------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def load_from_yaml(self):
        """
        [output] task_dictionary, hostname, database, username, pwd, port_id
        """
        global task_dictionary, hostname, database, username, pwd, port_id
        if self.file_path.is_file():
                with open(self.file_path) as yaml_file:
                        task_dictionary = yaml.safe_load(yaml_file)
                        hostname = str(task_dictionary["hostname"])  #'localhost'
                        database = str(task_dictionary["database"])  #'postgres'
                        username = str(task_dictionary["username"])  #'postgres'
                        pwd = str(task_dictionary["pwd"])            #'root'
                        port_id = str(task_dictionary["port"])       #5432
        else:
                print('agv configuration file missing.')
                exit(0)

    def dump_to_yaml(self, data):
        """
        [input] graph data
        [output] new json immediately for use.
        """
        if self.file_path.is_file():
                with open(self.file_path, "w") as outfile:
                        yaml.dump(data, outfile)
                self.load_from_yaml() # load new json immediately for use.
        else:
                print('agv configuration file missing.')
                exit(0)

##################################################
# ---------------------------------------------- #
#      ON START-UP: LOAD FLEET LIST AND MENU     #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- update_current_agv  ---------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def interactive_robot_fleet_startup(self):
        """
        [input] graph data
        [output] new json immediately for use.
        """
        # set the fleet_id and robot_id
        self.interactive_start_robot_fleet()
        # Create and start the timer
        # self.timer = Timer(10, self.on_timer(robot_id))  # 10 seconds
        # self.timer.start()
        if robot_id == 0:
                print("since no particular robot was seleted")
                print("exiting")
                return
        # Define the options
        options = [
                'fm_shutdown_trigger',
                'fm_robot_status_trigger',
                'fm_manual_control_trigger',
                'fm_send_task_request',
                'fm_clear_task_trigger',
                'fm_add_landmark_request',
                'fm_delete_landmark_request',
                'exit'
                ]
        # Print the options
        for i, option in enumerate(options, 1):
                print(f'{i}. {option}')
        # Ask for and validate the user's choice
        choice = input('Please enter the number of your choice: ')
        while not choice.isdigit() or not 1 <= int(choice) <= len(options):
                print('Invalid input. Please enter a number between 1 and', len(options))
                choice = input('Please enter the number of your choice: ')
        # Get the chosen option
        chosen_option = options[int(choice) - 1]

        # Call the corresponding function
        if chosen_option == 'fm_shutdown_trigger':
                answer = input('Alert! check yes or no for on or off? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')
                if answer == 'yes':
                        self.fm_shutdown_trigger(True)
                else:
                        self.fm_shutdown_trigger(False)

        elif chosen_option == 'fm_robot_status_trigger':
                answer = input('Alert! check yes or no for active or inactive? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')
                if answer == 'yes':
                        self.fm_robot_status_trigger(True)
                else:
                        self.fm_robot_status_trigger(False)

        elif chosen_option == 'fm_manual_control_trigger':
                answer = input('Alert! check yes or no to nav manually or not? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')
                if answer == 'yes':
                        self.fm_manual_control_trigger(True)
                        # Define the options
                        options = [
                                'left',
                                'right',
                                'forward',
                                'backward',
                                'stop',
                                'exit'
                        ]
                        # Print the options
                        for i, option in enumerate(options, 1):
                                print(f'{i}. {option}')
                        # Ask for and validate the user's choice
                        choice = input('Please enter the number of your choice: ')
                        while choice != 'exit':
                                while not choice.isdigit() or not 1 <= int(choice) <= len(options):
                                        print('Invalid input. Please enter a number between 1 and', len(options))
                                        choice = input('Please enter the number of your choice: ')
                                # Get the chosen option
                                chosen_option = options[int(choice) - 1]
                                # pass the chosen option to the db
                                if chosen_option == 'exit':
                                        self.fm_manual_control_drive('stop', robot_id, fleet_id)
                                        return
                                else:
                                        self.fm_manual_control_drive(chosen_option, robot_id, fleet_id)
                                choice = input('Please enter the number of your choice: ')
                else:
                        self.fm_manual_control_trigger(False)

        elif chosen_option == 'fm_send_task_request':

                # Define the valid options
                valid_loc_id_pattern = re.compile(r'^[A-Z][1-9]\d*|^[A-Z]10\d*$')

                # Ask for and validate from_loc_id
                from_loc_id = input('Please enter from_loc_id [options must conform to the pattern AnyAlphabet+<number>]: ')
                while not valid_loc_id_pattern.match(from_loc_id) or from_loc_id not in self.job_ids:
                        print('Invalid input. It does not conform to the AnyAlphabet+<number> pattern or not in : '+str( self.job_ids))
                        from_loc_id = input('Please enter from_loc_id [options must conform to the pattern AnyAlphabet+<number>] or exit: ')
                        if from_loc_id == 'exit':
                                return

                # Ask for and validate to_loc_id
                to_loc_id = input('Please enter to_loc_id [options must conform to the pattern AnyAlphabet+<number>]: ')
                while not valid_loc_id_pattern.match(to_loc_id) or to_loc_id not in self.job_ids:
                        print('Invalid input. It does not conform to the AnyAlphabet+<number> pattern or not in : '+str( self.job_ids))
                        to_loc_id = input('Please enter to_loc_id [options must conform to the pattern AnyAlphabet+<number>] or exit: ')
                        if to_loc_id == 'exit':
                                return

                # Ask for and validate task_name
                task_name = input('Please enter task_name [move, loop, charge, clean, transport]: ')
                while task_name not in self.job_types:
                        print('Invalid input. The task name must be one of the following: move, loop, charge, clean, transport.')
                        task_name = input('Please enter task_name [move, loop, charge, clean, transport] or exit: ')
                        if task_name == 'exit':
                                return

                # Ask for and validate task_priority
                task_priority = input('Please enter task_priority [low, high, medium]: ')
                while task_priority not in self.job_priorites:
                        print('Invalid input. The task priority must be one of the following: low, high, medium.')
                        task_priority = input('Please enter task_priority [low, high, medium] or exit: ')
                        if task_priority == 'exit':
                                return

                # Now you have valid inputs and you can proceed with your task
                print(f'from_loc_id: {from_loc_id}, to_loc_id: {to_loc_id}, task_name: {task_name}, task_priority: {task_priority}')

                answer = input('WARNING! Are you sure you want to continue with robot task? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')

                if answer == 'no':
                        if checkpoints != []:
                                checkpoints = []
                        print(" checkpoints now emptied: ", str(checkpoints))
                        return
                else:
                        self.fm_send_task_request(from_loc_id, to_loc_id, task_name, task_priority)

        elif chosen_option == 'fm_clear_task_trigger':
                answer = input('WARNING! Are you sure you want to clear/reset robot task? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')
                if answer == 'yes':
                        self.fm_clear_task_trigger(True)

        elif chosen_option == 'fm_add_landmark_request':
                # drive robot to the location you want to save as landmark.
                # use the interactive options to assign name and neighbour.

                # Ask for and validate station_type
                station_type = input('Please enter station_type from: '+str(self.station_type)+'.')
                while station_type not in self.station_type:
                        print('Invalid input. The station_type must be one of the following: '+str(self.station_type)+'.')
                        station_type = input('Please enter station_type : '+str(self.station_type)+'. or exit: ')
                        if station_type == 'exit':
                                return

                # Define the valid options
                valid_loc_id_pattern =  re.compile(r'^[A-Z][1-9]\d*|^[A-Z]10\d*$')
                # Ask for and validate from_loc_id
                input_loc_id = input('Please enter valid location/landmark id [options must conform to the pattern AnyAlphabet+<number>]: ')
                while not valid_loc_id_pattern.match(input_loc_id):
                        print('Invalid input. It does not conform to the AnyAlphabet+<number> pattern.')
                        input_loc_id = input('Please enter input_loc_id [options must conform to the pattern AnyAlphabet+<number>] or exit: ')
                        if input_loc_id == 'exit':
                                return

                # Ask for and validate from_loc_id
                print('ATTENTION! '+"Field 'Neighbors' cannot be left empty. Please choose appropriate connected waypoints.")

                # Define the valid pattern
                valid_loc_id_pattern = re.compile(r'^A-Z(,A-Z)*$')

                # Ask for and validate neighbor_loc_ids sample input_str = "A4,A2,A3,A6"
                neighbor_loc_ids = input('Please enter valid location/landmark id [options must conform to the pattern AnyAlphabet+<number>]: ')
                while not valid_loc_id_pattern.match(neighbor_loc_ids.replace(" ","")):
                        print('Invalid input. It does not conform to the AnyAlphabet+<number> pattern.')
                        neighbor_loc_ids = input('Please enter neighbor_loc_ids [options must conform to the pattern AnyAlphabet+<number>] or exit: ')
                        if neighbor_loc_ids == 'exit':
                                return

                self.fm_add_landmark_request(station_type, input_loc_id, neighbor_loc_ids)

        elif chosen_option == 'fm_delete_landmark_request':
                # Define the valid pattern
                valid_pattern = re.compile(r'^[A-Z][1-9]\d*|^[A-Z]10\d*$')
                # Ask for and validate the input
                input_str = input('Please enter a string of the form AnyAlphabet<number>,<word>: ')
                while not valid_pattern.match(input_str.replace(" ","")):
                        print('Invalid input. It does not conform to the AnyAlphabet<number>,<word> pattern.')
                        input_str = input('Please enter a string of the form AnyAlphabet<number>,<word> or exit: ')
                        if input_str == 'exit':
                                return

                self.fm_delete_landmark_request(input_str)

        elif chosen_option == 'exit':
                print("All of this was written and designed by Hazeezadebayo.")
                print("                                           Thank you.")
                return

##################################################
# ---------------------------------------------- #
#      ON START-UP: LOAD FLEET LIST AND MENU     #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- update_current_agv  ---------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def interactive_start_robot_fleet(self):
                """
                from Terminal interaction,
                [input] fleet_id, robot_id
                """
                global fleet_id, robot_id

                # get all fleet ids.
                self.fleet_ids =[]
                try:
                        self.cur.execute("SELECT DISTINCT fleet_id FROM table_robot") # ORDER BY something ascending descending etc
                except psycopg2.errors.UndefinedTable:
                        print('ATTENTION! The table "table_robot" does not exist.')
                        print('please startup a client robot first for a usable ui.')
                        exit(0)
                self.fleet_sql = self.cur.fetchall()
                for i in self.fleet_sql:
                        self.fleet_ids.append(i[0])

                # request for a particular fleet_id
                fleet_id = input('Please enter fleet_id '+str(self.fleet_ids)+': ')
                while fleet_id not in self.fleet_ids:
                        print('Invalid input...')
                        fleet_id = input('Please enter fleet id: ')

                self.fm_start_robot_fleet(fleet_id)


                answer = input('Alert! Do you want to assign a robot a task? [yes, no]: ')
                while answer not in self.decision:
                        print('Invalid input. The decision must be one of the following: yes, no.')
                        answer = input('Please enter decision [yes, no]: ')
                if answer == 'yes':
                        # Ask for a particular robot by robot_id
                        robot_id = input('Please enter int robot_id: '+str(self.robot_ids)+'.')
                        while robot_id not in self.robot_ids:
                                print('Invalid input. The robot id must be one of the following integers: '+str(self.robot_ids)+'.')
                                robot_id = input('Please enter robot id: ')
                        self.single_robot_view = True
                        self.on_timer(robot_id)
                else:
                        self.single_robot_view = False
                        self.on_timer()


    def fm_start_robot_fleet(self, f_id):
                """
                [API] call with fleet_id.
                pass a specific fleet id, then it populates to memory all robot from that fleet.
                it also pupulates what jobs/itinerary/landmarks are available to this specific fleet
                """

                global fleet_id
                fleet_id = f_id

                # get all fleet ids.
                self.fleet_ids =[]
                try:
                        self.cur.execute("SELECT DISTINCT fleet_id FROM table_robot") # ORDER BY something ascending descending etc
                except psycopg2.errors.UndefinedTable:
                        print('ATTENTION! The table "table_robot" does not exist.')
                        return
                self.fleet_sql = self.cur.fetchall()
                for i in self.fleet_sql:
                        self.fleet_ids.append(i[0])

                if fleet_id not in self.fleet_ids:
                        print("Invalid input. The fleet id must be one of the following integers:",str(self.fleet_ids))
                        return

                temp_type = None
                for i in range(0,len(task_dictionary["itinerary"])):
                        if task_dictionary["itinerary"][i]["fleet_id"] == fleet_id:
                                if (task_dictionary["itinerary"][i]["description"] == 'station_dock' or \
                                        task_dictionary["itinerary"][i]["description"] == 'charge_dock' or \
                                        task_dictionary["itinerary"][i]["description"] == 'elevator_dock' or \
                                                task_dictionary["itinerary"][i]["description"] == 'home_dock'):
                                        self.job_ids.append(task_dictionary["itinerary"][i]["loc_id"])

                                self.landmark_dict['x'].append(float(task_dictionary["itinerary"][i]["coordinate"][0]))
                                self.landmark_dict['y'].append(float(task_dictionary["itinerary"][i]["coordinate"][1]))
                                self.landmark_dict['t'].append(task_dictionary["itinerary"][i]["loc_id"])
                                if (task_dictionary["itinerary"][i]["description"] == 'charge_dock'): temp_type = 'green'
                                elif (task_dictionary["itinerary"][i]["description"] == 'home_dock'): temp_type = 'blue'
                                elif (task_dictionary["itinerary"][i]["description"] == 'station_dock'): temp_type = 'cyan'
                                elif (task_dictionary["itinerary"][i]["description"] == 'elevator_dock'): temp_type = 'orange'
                                else: temp_type = 'yellow'
                                self.landmark_dict['k'].append(temp_type)

                print("\nvalid landmark ids that can be used for jobs are:")
                print("[landmark_ids]: ", self.job_ids)

                # get all robots in this fleet_id
                self.robot_ids = []
                self.cur.execute("SELECT robot_id FROM table_robot WHERE fleet_id = %s;", (fleet_id,))
                self.robot_sql = self.cur.fetchall()
                for i in self.robot_sql:
                        self.robot_ids.append(str(i[0]))

                print("\nvalid robot ids that can be used here are:")
                print("[robot_ids]: ", self.robot_ids)

                # plot the world first. then we can plot the car/robot in the timed function.
                self.nav_map_obtained = False

##################################################
# ---------------------------------------------- #
#      on_timer callback ( AGV update)           #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ------------------------------- SINGLE AGV VISUALIZATION -------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def on_timer(self, r_id=0):
                """ # Define the function to be called by the timer : update plot """
                if r_id != 0:
                        global robot_id
                        robot_id = r_id
                        try:    # show robot with id robot_id in this fleet on the map.
                                self.cur.execute('SELECT * FROM table_robot WHERE robot_id = %s and fleet_id = %s;', (robot_id, fleet_id,))
                                for record in self.cur.fetchall():
                                        self.feedback_current_pose = str(record['current_pose']).split(','); # print(self.feedback_current_pose[0], self.feedback_current_pose[1], self.feedback_current_pose[2])
                                        self.feedback_notifications = str(record['notifications']).split(','); # print(self.feedback_notifications[0], self.feedback_notifications[1], self.feedback_notifications[2], self.feedback_notifications[3])
                                        self.feedback_last_checkpoint = str(record['last_checkpoint']).split(','); #
                                        self.feedback_checkpoints = str(record['checkpoints']).split(',');# print(len(self.feedback_checkpoints),"-route: ", self.feedback_checkpoints) # [A,B,C,D]
                                        self.feedback_agv_status = str(record['agv_status']).split(','); # print(self.feedback_agv_status)
                                        self.feedback_shutdown = str(record['shutdown']); # print(self.feedback_shutdown)
                                        self.feedback_controls = str(record['m_controls']).split(','); # print(self.feedback_controls[0], self.feedback_controls[1], self.feedback_controls[2], self.feedback_controls[3])
                                        self.feedback_model_config = str(record['model_config']).split(','); # print(self.controls[0], self.controls[1], self.controls[2], self.controls[3])
                                        self.feedback_landmarks = str(record['landmark']).split(','); #
                                        self.feedback_battery_status = str(record['battery_status']).split(','); #
                                        self.feedback_base_map_img = record['base_map_data']
                                        self.feedback_floor1_map_img = record['floor1_map_data']
                        except Exception as error:
                                print("[manager]:-on_timer-"+str([error])+".")
                else:
                        try:    # show all robot's in this fleet on the fleet map.
                                self.cur.execute("SELECT robot_id, base_map_data FROM table_robot WHERE fleet_id = %s;", (fleet_id,))
                                record = self.cur.fetchone()
                                if record is not None:
                                        r_id_ = str(record['robot_id'])
                                        self.feedback_base_map_img = record['base_map_data']
                        except Exception as error:
                                print("[manager]:-on_timer-"+str([error])+".")
                        print("[manager]:-map for robot with id: "+str(r_id_)+", in fleet: "+str(fleet_id)+", is selected for use.")
                # plot the world
                self.get_world()
                # plot the car
                self.nav_plot()
                # general: notify user if any problem in any fleet's robot
                self.check_notification_msgs()

# ---------------------------------------------- #
#      get_world  (WALLS AS OBSTACLES)           #
# ---------------------------------------------- #
# ----------------------------------------------------------------------------------------------------
# ----------------------------------  WORLD PLOT -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def get_world(self):
        if (self.nav_map_obtained == False) and (self.feedback_base_map_img is not None) and (self.feedback_base_map_img != 'None'):
                self.allobs_x,self.allobs_y = [],[]
                # img_path = "/colcon_ws/src/nav2_rosdevday_2021/nav2_rosdevday_2021/maps/my_cartographer_map.pgm"
                # src = cv2.imread(img_array, cv2.IMREAD_GRAYSCALE)
                # src = cv2.flip(src, 0)
                # Convert the memory object to bytes
                img_data = self.feedback_base_map_img.tobytes()
                img = Image.open(io.BytesIO(img_data))
                # Display the image
                # plt.imshow(img)
                # plt.show()
                # Open the image
                img_array = np.array(img)
                src = cv2.flip(img_array, 0)
                # Set threshold and maxValue
                thresh = 5 # 0
                maxValue = 255
                # Basic threshold example
                th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
                for i in range (dst.shape[0]): #traverses through height of the image  # height, width = src.shape[:2]
                        for j in range (dst.shape[1]): #traverses through width of the image
                                if dst[i][j] == 0:           # print('wall coordinates: {},{}'.format(i,j))
                                        self.allobs_x.append((i*0.05)-10.15)  # 12.3) - Kullar  # 10)-BV  # while scale fixed, map moves downward if increased.
                                        self.allobs_y.append((j*0.05)-9.8)   # 6.3)  - Kullar  # 10)-BV # while scale fixed, map moves backward if increased.
                self.nav_map_obtained = True
        else:
                print("[manager]:-No image found. please ensure that fleet_map_path in client is properly set.")
                return

# ---------------------------------------------- #
#   nav_plot.  (SHOW CURRENT AGV TRAJECTORY)     #
# ---------------------------------------------- #
# ----------------------------------------------------------------------------------------------------
# ------------------------------------  NAV PLOT -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def nav_plot(self):
        if (self.show_robot_animation is True) and (self.feedback_base_map_img is not None) and (self.feedback_base_map_img != 'None'):
                if self.single_robot_view:
                        if self.agv_track_dict['x'] != [] and self.agv_track_dict['y'] != []:  # it is not empty
                                # this is so that when its not moving it stops appending same coordinates to list.
                                if (float(self.agv_track_dict['x'][-1]) != float(self.feedback_current_pose[0])) and (float(self.agv_track_dict['y'][-1]) != float(self.feedback_current_pose[1])):
                                        self.agv_track_dict['x'].append(float(self.feedback_current_pose[0]))
                                        self.agv_track_dict['y'].append(float(self.feedback_current_pose[1]))
                        else:   # initially, it is empty. hence, append first...
                                self.agv_track_dict['x'].append(float(self.feedback_current_pose[0]))
                                self.agv_track_dict['y'].append(float(self.feedback_current_pose[1]))
                        if len(self.agv_track_dict['x']) == 45:
                                self.agv_track_dict['x'].pop(0)
                        if len(self.agv_track_dict['y']) == 45:
                                self.agv_track_dict['y'].pop(0)
                        # Clear the axes
                        self.ax.cla()
                        self.cx, self.cy = [], []
                        if self.feedback_checkpoints != ['A','A','A']:
                                for i in range(len(self.feedback_checkpoints)):
                                        if self.feedback_checkpoints[i] in checkpoints: # if 7 in a
                                                self.cx.append(agv_itinerary[checkpoints.index(self.feedback_checkpoints[i])][0]) #b=a.index(7)
                                                self.cy.append(agv_itinerary[checkpoints.index(self.feedback_checkpoints[i])][1])
                                self.ax.scatter(self.cx, self.cy,  marker="P",facecolor='red') # "*", "X", "s", "P" ## -- checkpoints
                        if len(self.landmark_dict['x']) != 0: # scatter([1,2,3],[4,5,6],color=['red','green','blue'])
                                self.ax.scatter(self.landmark_dict['x'], self.landmark_dict['y'],  marker="P", color=self.landmark_dict['k'])
                        for i in range(0, len(self.landmark_dict['x'])):
                                self.ax.annotate(self.landmark_dict['t'][i], (self.landmark_dict['x'][i], self.landmark_dict['y'][i]))
                        # plot walls/obstacle, basically plot the MAP
                        self.ax.scatter(self.allobs_y, self.allobs_x,  marker="X",facecolor='black') # "*", "X", "s", "P" ##
                        # plot the agv trajectory in orange, basically agv footprint
                        self.ax.plot(self.agv_track_dict['x'], self.agv_track_dict['y'], color='orange',linestyle='dashed', label="trajectory")
                        # plot the agv itself, basically a beautiful vehicle
                        self.plot_car(float(self.feedback_current_pose[0]), float(self.feedback_current_pose[1]), float(self.feedback_current_pose[2]), steer=0)
                        # set some figure properties
                        self.ax.axis("tight") # equal on off tight auto square scaled
                        self.ax.grid(True)
                        if (self.feedback_agv_status[0] == 'idle') or (self.feedback_agv_status[0] == 'inactive') or (self.feedback_agv_status[0] == 'active'):
                                self.ax.set_title("AGV:" + str(robot_id)  + ", STATUS:" + str(self.feedback_agv_status[0])) # str(round(state.v * 3.6, 2))
                        plt.draw()
                        # plt.show() # ---> blocking call. code will pass only if window is closed
                        plt.pause(0.001)  # Non-blocking call. pause to update the figure
                else:
                        # initialize the world into memory
                        try:
                                self.cur.execute('SELECT current_pose FROM table_robot WHERE fleet_id = %s;', (fleet_id,))
                                for record in self.cur.fetchall():
                                        # Clear the axes
                                        self.ax.cla()
                                        self.feedback_all_poses = str(record['current_pose']).split(',')
                                        # plot walls/obstacle, basically plot the MAP
                                        self.ax.scatter(self.allobs_y, self.allobs_x,  marker="X",facecolor='black') # "*", "X", "s", "P" ##
                                        # plot the agvs, basically a beautiful vehicle
                                        self.plot_car(float(self.feedback_all_poses[0]), float(self.feedback_all_poses[1]), float(self.feedback_all_poses[2]), steer=0)
                                        plt.draw()
                                        # plt.show() # ---> blocking call. code will pass only if window is closed
                                        plt.pause(0.001)  # Non-blocking call. pause to update the figure
                        except Exception as error:
                                print("[manager]:-nav_plot-"+str([error])+".")

# ---------------------------------------------- #
#      PLOT CAR:                                 #
# ---------------------------------------------- #
# ----------------------------------------------------------------------------------------------------
# ------------------------------------  car PLOT -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

        outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                                [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

        fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                                [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                        [-math.sin(steer), math.cos(steer)]])

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += WB
        fl_wheel[0, :] += WB

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        self.ax.plot(np.array(outline[0, :]).flatten(),np.array(outline[1, :]).flatten(), truckcolor)
        self.ax.plot(np.array(fr_wheel[0, :]).flatten(),np.array(fr_wheel[1, :]).flatten(), truckcolor)
        self.ax.plot(np.array(rr_wheel[0, :]).flatten(),np.array(rr_wheel[1, :]).flatten(), truckcolor)
        self.ax.plot(np.array(fl_wheel[0, :]).flatten(),np.array(fl_wheel[1, :]).flatten(), truckcolor)
        self.ax.plot(np.array(rl_wheel[0, :]).flatten(),np.array(rl_wheel[1, :]).flatten(), truckcolor)
        self.ax.plot(x, y, "*")

##################################################
# ---------------------------------------------- #
#      SEND TASK REQUEST                         #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- fm_send_task_request  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_send_task_request(self, from_loc_id, to_loc_id, task_name, task_priority):
                """
                Args:
                        from_loc_id (_type_): _description_
                        to_loc_id (_type_): _description_
                        task_name (_type_): _description_
                        task_priority (_type_): _description_
                """
                global checkpoints, landmark, agv_itinerary, robot_id, fleet_id

                self.request_tasks(from_loc_id, to_loc_id, task_name, task_priority)

                if robot_id == 0:
                        print("[manager]:-ATTENTION! Robot is not selected. Please choose robot.")
                        return

                if len(checkpoints) != 0:
                        if task_cleared == True:
                                self.feedback_agv_status[0] = 'active'
                                try: # A,B,C,D,E,F - checkpoints
                                        self.cur.execute('UPDATE table_robot SET agv_status = %s, checkpoints = %s, landmark = %s, agv_itinerary = %s WHERE robot_id = %s and \
                                        fleet_id = %s', (','.join(self.feedback_agv_status),','.join(checkpoints),','.join(landmark), agv_itinerary, robot_id, fleet_id,))
                                        self.conn.commit()
                                except Exception as error:
                                        print("[manager]:-fm_send_task_request-"+str([error])+".")
                        else:
                                print('[manager]:-ATTENTION! Task not cleared. Hence, not assigned.')

# ---------------------------------------------- #
#     ON TASK REQUEST                            #
# ---------------------------------------------- #
# ----------------------------------------------------------------------------------------------------
# -------------------------- TaskScheduler: request_tasks  -------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def request_tasks(self, from_loc_id, to_loc_id, task_name, task_priority):
        # from_loc_id | to_loc_id | task_name | task_priority
        """
        given a selected job id locate job information and itinerary and assign task to robot.
        """

        global checkpoints, landmark, agv_itinerary, task_dictionary, robot_id, task_cleared
        checkpoints, agv_itinerary, landmark = [], [], []
        task_cleared = False

        # ROBOT_ID and FLEET_ID have already been assigned immediately a robot is selected. Hence, the plot.

        if robot_id == 0:
              print("[manager]:-Robot is not selected. Please choose robot.")
              return

        if from_loc_id == to_loc_id and (task_name != 'charge' and task_name != 'move'):
              print("[manager]:-'load' location cannot be equal to 'unload' location.")
              return

        if (from_loc_id not in self.job_ids) or (to_loc_id not in self.job_ids):
                print("[manager]:-only ids within this fleet ["+str(fleet_id)+"] form a valid task request.")
                return


        shortest_distance_found = 10000000000000000000000
        loc_node_owner = None #  just for initialization to hold values.
        home_dock_loc_ids = []
        charge_dock_loc_ids = []

        #  1 landmark (charge) - charge | 3 landmarks (pick, drop, home > 1) - transport | 2 landmarks (pick, drop) - loop

        print("[manager]:-robot at: ", self.feedback_current_pose[0], self.feedback_current_pose[1], self.feedback_current_pose[2])
        for i in range(0,len(task_dictionary["itinerary"])):
                if task_dictionary["itinerary"][i]["fleet_id"] == fleet_id:
                        # build a list of all home_dock stations in this fleet
                        if task_dictionary["itinerary"][i]["description"] == 'home_dock':
                              home_dock_loc_ids.append(task_dictionary["itinerary"][i]["loc_id"])
                        # build a list of all charge_dock stations in this fleet
                        if task_dictionary["itinerary"][i]["description"] == 'charge_dock':
                              charge_dock_loc_ids.append(task_dictionary["itinerary"][i]["loc_id"])
                        # check if intended start location is in the fleet we belong
                        # handle case where there is a barrier for example, and shortest distance is not really the real world shortest distance.
                        coordinate = task_dictionary["itinerary"][i]["coordinate"]
                        d = math.sqrt((float(coordinate[0]) - float(self.feedback_current_pose[0]))**2 + (float(coordinate[1]) - float(self.feedback_current_pose[1]))**2)
                        if d < shortest_distance_found:
                                shortest_distance_found = d
                                loc_node_owner = task_dictionary["itinerary"][i]["loc_id"]
                                print("[manager]:-found shortestest dist: ", shortest_distance_found," with loc: ", loc_node_owner)

        imaginary_checkpoints = []
        real_checkpoints = []
        temp_dock_list = []
        landmark = []
        checkpoints = []

        if task_name == 'charge' or task_name == 'move':
                if task_name == 'charge':
                        # for the charge task, just gather all the charge stations and build a path linking all of them
                        landmark = [task_priority, task_name]
                        if len(charge_dock_loc_ids) != 0:
                                # for each charge dock station we have
                                for i in range(0,len(charge_dock_loc_ids)):
                                        if i == 0:
                                                # obtain a path from the robot to the first charge dock station
                                                temp_dock_list.append(self.shortest_path(graph, loc_node_owner, charge_dock_loc_ids[i]))
                                        else:
                                                # from i = 1,
                                                # we wanna get path from one charge dock to the next charge dock
                                                # essentially, if the first charge dock is occupied, we can move to the next
                                                temp_dock_list.append(self.shortest_path(graph, charge_dock_loc_ids[i-1], charge_dock_loc_ids[i]))
                                        # add the charge dock in the order we have analyzed.
                                        landmark.append(charge_dock_loc_ids[i])
                                # sum all the checkpoints
                                # and make sure there is no repeated start and end checkpoints.
                                for i in range(len(temp_dock_list)):
                                        if i == 0:
                                                checkpoints += temp_dock_list[i]
                                        elif temp_dock_list[i-1][-1] == temp_dock_list[i][0]:
                                                checkpoints += temp_dock_list[i][1:]
                                        else:
                                                checkpoints += temp_dock_list[i]
                # if its move, check if the place the robot is closest to,
                # which is where we have assumed is the start point/node,
                # check if it is also the target node,
                # if it is not, then get the path that leads there.
                if task_name == 'move':
                        if loc_node_owner != to_loc_id:
                                checkpoints = self.shortest_path(graph, loc_node_owner, to_loc_id)
                                landmark = [task_priority, task_name, to_loc_id]
                else:
                      pass

        elif task_name == 'transport' or task_name== 'loop':
                if loc_node_owner != from_loc_id:
                        imaginary_checkpoints = self.shortest_path(graph, loc_node_owner, from_loc_id)
                real_checkpoints = self.shortest_path(graph, from_loc_id, to_loc_id)
                # Check if the last element of lst_1 is the same as the first element of lst_2
                if len(imaginary_checkpoints) > 0:
                        if imaginary_checkpoints[-1] == real_checkpoints[0]:
                                checkpoints = imaginary_checkpoints + real_checkpoints[1:]
                else:
                        # If they are different, output both lists concatenated
                        checkpoints = real_checkpoints

                if task_name == 'loop':
                        landmark = ['high', task_name, from_loc_id, to_loc_id]
                        return_checkpoints = self.shortest_path(graph, to_loc_id, from_loc_id)
                        if len(return_checkpoints) != 0:
                                if checkpoints[-1] == return_checkpoints[0]:
                                        checkpoints = checkpoints + return_checkpoints[1:]

                if task_name == 'transport':
                        landmark = [task_priority, task_name, from_loc_id, to_loc_id]
                        if len(home_dock_loc_ids) != 0:
                                for i in range(0,len(home_dock_loc_ids)):
                                        if i == 0:
                                                temp_dock_list.append(self.shortest_path(graph, to_loc_id, home_dock_loc_ids[i]))
                                        else:
                                                temp_dock_list.append(self.shortest_path(graph, home_dock_loc_ids[i-1], home_dock_loc_ids[i]))
                                        landmark.append(home_dock_loc_ids[i])

                                temp_dock_checkpoints = []
                                for i in range(len(temp_dock_list)):
                                        if i == 0:
                                                temp_dock_checkpoints += temp_dock_list[i]
                                        elif temp_dock_list[i-1][-1] == temp_dock_list[i][0]:
                                                temp_dock_checkpoints += temp_dock_list[i][1:]
                                        else:
                                                temp_dock_checkpoints += temp_dock_list[i]

                                if len(temp_dock_checkpoints) != 0:
                                        if checkpoints[-1] == temp_dock_checkpoints[0]:
                                                checkpoints = checkpoints + temp_dock_checkpoints[1:]

        elif task_name == "clean":
        # if the task is clean, we do not have any landmark because
        # the coverage path planner will generate path on the map to be traversed.
        # and this path does not necessarily imply the normal routes traversed by other robots within same fleet.
        # truth is! if you have more than 1 cleaning robot in your fleet, then this might be a problem as probability
        # for a robot stand-off becomes high.
        # because idea is;
        #       once its a cleaning task we arent really doing anything other than monitoring the robots.
        #       and checking if its gonna collide with anything and then stop it till the obstacle leaves.
        #       we can not alter a path for which we did not create.
                landmark = ['low', task_name, "unknown", "unknown"]
                checkpoints = ['A', 'B', 'A'] # default no-task structure
                agv_itinerary = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]] # '[x,y,z,w],[x,y,z,w],[x,y,z,w]' --> 'A', 'A', 'A'
        else:
              pass

        if len(checkpoints) != 0:
                print("[manager]:- writing path into text... \n")
                print("[manager]:-",str(checkpoints))
                print("[manager]:- ")
                # checkpoints -- alphabets or location ids
                # obtain agv itinerary -- coordinate
                for x in range(0,len(checkpoints)):
                        for y in range(0,len(task_dictionary["itinerary"])):
                                if task_dictionary["itinerary"][y]["loc_id"] == checkpoints[x]:
                                        agv_itinerary.append(task_dictionary["itinerary"][y]["coordinate"])
        # landmark -- [task priority, task type or name, landmarks]
        # do predecessor landmark stuff here;
        if landmark != []:
               if (landmark[1] == 'loop') or (landmark[1] == 'transport'):
                       # landmark[2] here is pick. recall 0 is priority, 1 is type, 2 is pick, 3 is drop,
                       # for transport, 4 is home, but for loop, it goes back to pick.
                       # so select the alphabet that represent the location after landmark pick.
                       # i am using + for the first instance because i am sure when it wants to pull out
                       # no pun intended, it must pass by the gate. however, the start point might have been
                       # precisely the pick location. i dont know. so the would have been no predecessor in that
                       # case. you know? gotta be smart!
                        landmark[2] = checkpoints[checkpoints.index(landmark[2]) + 1]+'_'+landmark[2]
                        # select the alphabet that represent the location before landmark drop.
                        landmark[3] = checkpoints[checkpoints.index(landmark[3]) - 1]+'_'+landmark[3]
                        if landmark[1] == 'transport':
                                # 4 upwards is home for transport we need to get all predecessor/gate locations too.
                                for idx, elem in enumerate(landmark[4:]): # Here, `elem` is the current element in the loop and `idx` is its index within the list # Do something with `prev_elem`
                                        landmark[4+idx] = checkpoints[checkpoints.index(elem) - 1]+'_'+landmark[4+idx]
               elif landmark[1] == 'move':
                        landmark[2] = checkpoints[checkpoints.index(landmark[2]) - 1]+'_'+landmark[2]
               elif (landmark[1] == 'charge'):
                        for idx, elem in enumerate(landmark[2:]): # Here, `elem` is the current element in the loop and `idx` is its index within the list
                                landmark[2+idx] = checkpoints[checkpoints.index(elem) - 1]+'_'+landmark[2+idx]
        else:
               landmark = ['none']

        print("[manager]:- checkpoints to pass: ", checkpoints)
        print("[manager]:- landmark to visit: ", landmark)

        # schedule task
        self.task_scheduled(task_name)

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------

    def task_scheduled(self, task_name):
        global task_cleared
        # Find free robots or available robots not currently performing any task from table_robots
        try: # try to open a connection yeah. but if it closes abruptly, # connection will never be closed hence, move 'close()' to finally
            #select_script = '''SELECT * FROM table_robot''' # '''SELECT id,name FROM table_robot'''
            select_script = "SELECT * FROM table_robot WHERE robot_id = %s;"
            select_id = (robot_id,)
            self.cur.execute(select_script, select_id)
            # print(cur.fetchall()) # everyone gets printed on one line.
            for record in self.cur.fetchall(): #print(record) # now everyone gets its own line e.g [2, 'Robin', 22500, 'D1'] [3, 'Xavier', 19500, 'D2']
                checkp_pos_ = str(record['checkpoints']).split(',');# print(len(self.checkpoints),"-route: ", self.checkpoints) # [A,B,C,D]
                agv_stat_ = str(record['agv_status']).split(',')
                battery_stat_ = str(record['battery_status']).split(',')
        except Exception as error: # show the exceptions basically as errors.
            print("[manager]:-task_scheduled-"+str([error])+".")

        try:
                if ((checkp_pos_[0] == checkp_pos_[1] == checkp_pos_[2]) or (agv_stat_[0] == 'idle')):
                        # robot is available
                        if (float(battery_stat_[-1]) > 35) or (float(battery_stat_[-1]) <= 35 and task_name == 'charge'):
                                task_cleared = True
                        elif (float(battery_stat_[-1]) <= 35) and (task_name != 'charge'):
                                task_cleared = False
                                self.available_robots = self.suggest_robot()
                                if len(self.available_robots) > 0:
                                        print("[manager]:-Robot battery percent is less than 35%. Only 'charge' task is allowed. Please consider robot "+str(self.available_robots[0])+".")
                                else:
                                        print('[manager]:-ATTENTION! '+" battery percent is less than 35%. only 'charge' task is allowed.")
                else:
                        # robot is not available
                        task_cleared = False
                        self.available_robots = self.suggest_robot()
                        if len(self.available_robots) > 0:
                                print('[manager]:-ATTENTION! '+'Robot unavailable. Robot is currently active. Please cancel current task and reassign or consider robot '+str(self.available_robots[0])+'.')
                        else:
                                print('[manager]:-ATTENTION! '+'Robot unavailable. Robot is currently active. Please cancel current task and reassign.')
        except ValueError:
                print("[manager]:-task_scheduled- could not convert battery stat to float.")

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------

    def suggest_robot(self):
        available_robots = [] # - Assign task to any available robot. but in a while loop.
        try:
                self.cur.execute("SELECT * FROM table_robot WHERE fleet_id = %s;", (fleet_id,))
                for record in self.cur.fetchall():
                        r_id_ = str(record['robot_id']).split(',')
                        checkp_p_ = str(record['checkpoints']).split(','); # [A,B,C,D]
                        agv_s_ = str(record['agv_status']).split(',')
                        battery_s_ = str(record['battery_status']).split(',');  # (str(self.bat_v)+","+str(self.bat_t)+","+str(self.bat_c)+","+str(self.bat_p)
                        if ((checkp_p_[0] == checkp_p_[1] == checkp_p_[2])) or ((agv_s_[0] == 'idle')):
                                # this robot is available print("battery_s_[-1]: ", float(battery_s_[-1]) )
                                if (float(battery_s_[-1]) > 35):
                                        available_robots.append(r_id_)
                                        break
        except (ValueError, Exception) as error:
                print("[manager]:-suggest_robot-"+str([error])+".")

        return available_robots

        # ----------------------------------------------------------------------------------------------------
        # ---------------------------- PathScheduler: dijkstra  ----------------------------------------------
        # ----------------------------------------------------------------------------------------------------

    def dijkstra(self, graph, start):
                dist = {node: float('inf') for node in graph}
                dist[start] = 0
                heap = [(0, start)]
                while heap:
                        (d, node) = heapq.heappop(heap)
                        if d > dist[node]:
                                continue
                        for neighbor, weight in graph[node]:
                                new_dist = dist[node] + weight
                                if new_dist < dist[neighbor]:
                                        dist[neighbor] = new_dist
                                        heapq.heappush(heap, (new_dist, neighbor))
                return dist

        # ----------------------------------------------------------------------------------------------------
        # ---------------------------- PathScheduler: shortest_path ------------------------------------------
        # ----------------------------------------------------------------------------------------------------

    def shortest_path(self, graph, start, goal):
                distances = self.dijkstra(graph, start)
                if goal not in distances:
                        return None
                paths = {start: [start]}
                heap = [(0, start)]
                while heap:
                        (d, node) = heapq.heappop(heap)
                        if node == goal:
                                return paths[node]
                        for neighbor, weight in graph[node]:
                                new_dist = distances[node] + weight
                                if neighbor not in paths or (new_dist < distances[neighbor] and len(paths[node]) + 1 <= len(paths[neighbor])):
                                        distances[neighbor] = new_dist
                                        paths[neighbor] = paths[node] + [neighbor]
                                        heapq.heappush(heap, (new_dist, neighbor))
                return None

##################################################
# ---------------------------------------------- #
#     LANDMARKING: SAVE CURRENT LOCATION         #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ------------------------------ fm_add_landmark_request  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_add_landmark_request(self, station_type, input_loc_id, neighbor_loc_ids): # limit by graph size, make only numeric
        if robot_id == 0:
                print('[manager]:-ATTENTION! '+"Robot is not selected. Please choose robot.")
                return

        if len(self.feedback_current_pose) < 1:
               return

        self.o_q = self.euler_to_quaternion(float(self.feedback_current_pose[2]), 0.0, 0.0)
        x = round(float(self.feedback_current_pose[0]), 2)
        y = round(float(self.feedback_current_pose[1]), 2)
        z = round(float(self.o_q[2]), 2)
        w = round(float(self.o_q[3]), 2)

        print("[manager]:-location or landmark to save: \n")
        print("[manager]:-x: "+str(x)+", y: "+str(y)+", z: "+str(z)+", w: "+str(w))
        print("[manager]:- ")

        # Define the regular expression pattern
        pattern = re.compile(r'^[A-Z][1-9]\d*|^[A-Z]10\d*$')
        # Check if input_loc_id matches the pattern
        if not pattern.match(input_loc_id):
                print('[manager]:-ATTENTION! ' + 'Location Number 0 cannot be chosen. Start numbering from 1.\
                        Only single capital lettered alphabets can be used. E.g. A1, E54 etc.')
                return
        else: # check if its already in the landmarks
                for i in range(0,len(task_dictionary["itinerary"])):
                        if task_dictionary["itinerary"][i]["fleet_id"] == fleet_id:
                                if task_dictionary["itinerary"][i]["loc_id"] == input_loc_id:
                                        print('[manager]:-ATTENTION! '+'Location ID already exists.')
                                        return

        # choose landmark neighbours for edge list
        # input_str = "A4,C2,C3,A6, D2,A1, A4"
        # remove spaces and split the string into a list of edges
        edges = neighbor_loc_ids.split(",")
        # define the regular expression pattern
        pattern = re.compile(r'^[A-Z]\d+$')
        # check if each element matches the pattern
        for element in edges:
                if not pattern.match(element.strip()):  # strip() is used to remove leading/trailing spaces
                        print('[manager]:-ATTENTION! ' + "Invalid element in 'Neighbors' field: " + element + ".")
                        return
                if element not in self.landmark_dict['t']:
                        print('[manager]:-ATTENTION! ' + "Element " + element + " in 'Neighbors' field not a landmark.")
                        return

        self.add_location_db_cmd(edges, x, y, w, z, station_type, input_loc_id, fleet_id)

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------

    def add_location_db_cmd(self, edges, x, y, w, z, loc_type, node, fleet_id):
        global task_dictionary

        temp_dict = {   "loc_id": node,
                        "fleet_id": fleet_id,
                        "description": loc_type,
                        "coordinate": [x, y, w, z]   }
        # more like if node == 'A1': then edge_list = [('A2', 1.0)] <-- just to initialize or prime the edge list.
        # the A2 will subsequently be updated with its cost once we actually save an A2 node.
        # Again, users need to name the nodes alphabetically and serially
        edge_list = []
        # Check if node matches the pattern of a single capital letter followed by one or more digits
        if re.match(r'^[A-Z]\d+$', node):
                # Extract the letter and the number from the node
                letter = node[0]
                number = int(node[1:])
                # Create the next node by incrementing the number
                next_node = f'{letter}{number + 1}'
                edge_list = [(next_node, 1.0)]
        else:
                for i in range(0,len(task_dictionary["itinerary"])):
                        if task_dictionary["itinerary"][i]["fleet_id"] == fleet_id:
                                # we need to get the coordinates of the neighbours listed
                                if task_dictionary["itinerary"][i]["loc_id"] in list(edges):
                                        # lets set it to coordinate yeah
                                        coordinate = task_dictionary["itinerary"][i]["coordinate"]
                                        # obtain the distance from this new landmark/location to the neighbour and use it as the cost.
                                        d = round(math.sqrt((float(coordinate[0]) - x)**2 + (float(coordinate[1]) - y)**2), 2)
                                        # update the new landmarks edge list with the (neighbour,cost)
                                        edge_list.append((task_dictionary["itinerary"][i]["loc_id"],d))
        # update the edges and cost of the listed neighbours
        self.add_node_and_edges(node, edge_list)
        # convert the graph to the desired YAML format update the 'graph' entity in the dictionary
        task_dictionary['graph'] = {} # clear the old graph
        for node, edges in graph.items():
                yaml_edges = []
                for edge in edges:
                        yaml_edges.append([edge[0], edge[1]])
                task_dictionary['graph'][node] = yaml_edges
        # Update the 'nodes' to include the recently added node
        task_dictionary["itinerary"].append(temp_dict)
        # Write the updated YAML back to the file
        self.dump_to_yaml(task_dictionary)
        # give the computer some time to breathe
        time.sleep(0.5)
        print('[manager]:-Add landmark completed successfully! '+'modify edges in the output config file. [yaml]!')

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------

    def add_node_and_edges(self, node, edges):
        global graph
        # print("node: ", node); print("edges: ", edges); print("graph: ", graph)
        # add the node to the graph, if not already present
        if node not in graph:
                graph[node] = []
        for edge in edges:
                # split the tuple into neighbor and cost
                neighbor, cost = edge
                cost = float(cost)
                # check if edge already exists
                existing_edge = None
                for i, n in enumerate(graph[node]):
                        if n[0] == neighbor:
                                existing_edge = i
                                break
                # if edge exists, update the cost
                if existing_edge is not None:
                        graph[node][existing_edge][1] = cost
                        # update the neighbor's edge as well
                        for i, n in enumerate(graph[neighbor]):
                                if n[0] == node:
                                        graph[neighbor][i][1] = cost
                # if edge does not exist, add the edge
                else:
                        graph[node].append([neighbor, cost])
                        # add the node to the neighbor's neighbors, if not already present
                        if neighbor not in graph:
                                graph[neighbor] = set()
                        if node not in [n[0] for n in graph[neighbor]]:
                                graph[neighbor].add((node, cost))

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

##################################################
# ---------------------------------------------- #
#     CLEAR/RESET TASK                           #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- fm_clear_task_trigger -------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_clear_task_trigger(self, trigger:bool):
        if robot_id == 0:
                print('[manager]:-ATTENTION! '+ "Robot is not selected. Please choose robot.")
                return

        if self.feedback_checkpoints[0] == 'A' or  self.feedback_checkpoints[0] == 'A' or self.feedback_checkpoints[0] == 'A':
               print('[manager]:-ATTENTION! '+ "Robot not currently executing any task. Operation not performed.")
               return

        if trigger is True:
                self.clear_task_db_cmd('reset')
                time.sleep(2.0)
                self.clear_task_db_cmd('inactive')

    def clear_task_db_cmd(self, whattodo):
                try:
                        self.cur.execute('UPDATE table_robot SET agv_status = %s, last_checkpoint = %s, checkpoints = %s WHERE robot_id = %s and \
                                fleet_id = %s', (whattodo+',none,x,y,z,w','unknown','A,A,A',robot_id,fleet_id,))
                        self.conn.commit()
                except Exception as error:
                        print("[manager]:-clear_task_db_cmd-"+str([error])+".")

##################################################
# ---------------------------------------------- #
#   ROBOT STATUS  (ACTIVE OR INACTIVE)           #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# --------------------------------- fm_robot_status_trigger ------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_robot_status_trigger(self, trigger:bool):
        if robot_id == 0:
                print('[manager]:-ATTENTION! Robot is not selected. Please choose robot.')
                return
        try:
                self.cur.execute('SELECT * FROM table_robot WHERE robot_id = %s and fleet_id = %s;', (robot_id, fleet_id,))
                for record in self.cur.fetchall():
                        _agv_stat = str(record['agv_status']).split(',')
                        if (trigger is False) and (_agv_stat[0] == 'active'):
                                self.robot_status_db_cmd('inactive', robot_id, fleet_id)
                        elif (trigger is True) and (_agv_stat[0] == 'inactive'):
                                self.robot_status_db_cmd('active', robot_id, fleet_id)
                        else:
                                pass
        except Exception as error:
                print("[manager]:-fm_robot_status_trigger-"+str([error])+".")

    def robot_status_db_cmd(self, agv_stat, robot_id, fleet_id):  # active/inactive - agv_stat
        self.feedback_agv_status[0] = agv_stat
        try:
            self.cur.execute('UPDATE table_robot SET agv_status = %s WHERE robot_id = %s and fleet_id = %s', (','.join(self.feedback_agv_status), robot_id, fleet_id,))
            self.conn.commit()
            print('[manager]:-Robot Status State: ', str(agv_stat))
        except Exception as error:
            print("[manager]:-robot_status_db_cmd-"+str([error])+".")

##################################################
# ---------------------------------------------- #
#     MANUAL CONTROL                             #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- send_control_nav  -----------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_manual_control_trigger(self, trigger:bool):
        if robot_id == 0:
                print('[manager]:-ATTENTION! Robot is not selected. Please choose robot.')
                return
        try:
                self.cur.execute('SELECT * FROM table_robot WHERE robot_id = %s and fleet_id = %s;', (robot_id, fleet_id,)) # ROBOT_ID
                for record in self.cur.fetchall():
                        txt2 = str(record['m_controls']); self.m_controls = txt2.split(',')
                        if (trigger is False) and self.m_controls[0] == 'yes':
                                self.manual_cntrl_db_cmd('no', robot_id, fleet_id)
                                print('[manager]:-Success! manual drive mode is turned off.')
                        elif (trigger is True) and self.m_controls[0] == 'no':
                                self.feedback_agv_status[0] = 'active'
                                self.cur.execute('UPDATE table_robot SET agv_status = %s, checkpoints = %s, last_checkpoint = %s WHERE robot_id = %s and \
                                        fleet_id = %s', (','.join(self.feedback_agv_status),'A,A,A','unknown',robot_id,fleet_id,))
                                self.conn.commit()
                                self.manual_cntrl_db_cmd('yes', robot_id, fleet_id)
        except Exception as error:
                print("[manager]:-fm_manual_control_trigger-"+str([error])+".")

    def fm_manual_control_drive(self, k_pressed, robot_id, fleet_id):
        return self.manual_cntrl_db_cmd(k_pressed, robot_id, fleet_id)

    def manual_cntrl_db_cmd(self, k_pressed, robot_id, fleet_id):  # yes False, False, False, False -- on/off left forward right stop - m_controls
        if k_pressed == 'yes' or k_pressed == 'no':
                if k_pressed == 'yes' : self.control_command = 'yes,False,False,False,False,False'
                elif k_pressed == 'no': self.control_command = 'no,False,False,False,False,False'
                try:
                        self.cur.execute('UPDATE table_robot SET m_controls = %s WHERE robot_id = %s and fleet_id = %s', (self.control_command, robot_id, fleet_id,))
                        self.conn.commit()
                except Exception as error:
                        print("[manager]:-manual_cntrl_db_cmd-1-"+str([error])+".")
        else: # Call the corresponding function
                try:
                        self.cur.execute('SELECT m_controls FROM table_robot WHERE robot_id = %s and fleet_id = %s;', (robot_id, fleet_id,)) # ROBOT_ID
                        for record in self.cur.fetchall():
                                txt2 = str(record['m_controls'])
                                self.m_controls = txt2.split(',')
                                if self.m_controls[0] == 'yes':
                                        if k_pressed == 'left': self.control_command       = 'yes,True,False,False,False,False'
                                        elif k_pressed == 'forward': self.control_command  = 'yes,False,True,False,False,False'
                                        elif k_pressed == 'right': self.control_command    = 'yes,False,False,True,False,False'
                                        elif k_pressed == 'stop': self.control_command     = 'yes,False,False,False,True,False'
                                        elif k_pressed == 'backward': self.control_command = 'yes,False,False,False,False,True'
                                        self.cur.execute('UPDATE table_robot SET m_controls = %s WHERE robot_id = %s and fleet_id = %s', (self.control_command, robot_id, fleet_id,))
                                        self.conn.commit()
                except Exception as error:
                        print("[manager]:-manual_cntrl_db_cmd-2-"+str([error])+".")

##################################################
# ---------------------------------------------- #
#   SHUTDOWM  (ON OR OFF)                        #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ---------------------------------- fm_shutdown_trigger  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_shutdown_trigger(self, trigger:bool):
        try:
                self.cur.execute('SELECT * FROM table_robot WHERE robot_id = %s and fleet_id = %s;', (robot_id, fleet_id,))
                for record in self.cur.fetchall():
                        if (trigger is True) and (str(record['shutdown']) == 'no'):
                                self.shutdown_db_cmd('yes', robot_id, fleet_id)
                        elif (trigger is False) and (str(record['shutdown']) == 'yes'):
                                self.shutdown_db_cmd('no', robot_id, fleet_id)
                        else:
                                pass
        except Exception as error:
                print("[manager]:-fm_shutdown_trigger-"+str([error])+".")

    def shutdown_db_cmd(self, shutdown, robot_id, fleet_id):  # yes/no - shutdown
        try:
            self.cur.execute('UPDATE table_robot SET shutdown = %s WHERE robot_id = %s and fleet_id = %s;', (shutdown, robot_id, fleet_id,))
            self.conn.commit()
            print('[manager]:-Robot Power State-', str(shutdown))
        except Exception as error:
            print("[manager]:-shutdown_db_cmd-"+str([error])+".")


##################################################
# ---------------------------------------------- #
#      SETTINGS:  (Delete landmark)              #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ----------------------------------  Delete landmark ------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    def fm_delete_landmark_request(self, str_location_id_comma_fleet):

        global task_dictionary

        parts = str_location_id_comma_fleet.split(',')
        loc_, fleet_ = parts[0], parts[1]

        if (loc_ == None) or (fleet_ == None):
                print("[manager]:-ATTENTION- Invalid element in 'delete landmark' field.")
                return

        for i in range(0,len(task_dictionary["itinerary"])):
                if task_dictionary["itinerary"][i]["fleet_id"] == fleet_:
                        if task_dictionary["itinerary"][i]["loc_id"] == loc_:
                                del task_dictionary["itinerary"][i]
                                break

        if loc_ in task_dictionary["graph"]:
                del task_dictionary["graph"][loc_]

        for node, edges in task_dictionary['graph'].items():
                task_dictionary['graph'][node] = [edge for edge in edges if edge[0] != loc_]

        self.dump_to_yaml(task_dictionary)

        time.sleep(0.5)


##################################################
# ---------------------------------------------- #
#      SEND SMS NOTOFICATION                     #
# ---------------------------------------------- #
##################################################
# ----------------------------------------------------------------------------------------------------
# ----------------------------------- check_notification_msgs  ---------------------------------------
# ----------------------------------------------------------------------------------------------------
    def check_notification_msgs(self):
        """ twilio_server: -- startup msgs -- system error msg -- dock related/occupied msg -- reverse/recovery msg """
        try:
                self.cur.execute("SELECT notifications, robot_id FROM table_robot WHERE fleet_id = %s;", (fleet_id,))
                for record in self.cur.fetchall():
                        notifications_ = str(record['notifications']).split(',')
                        r_id_ = str(record['robot_id'])
                        # print("notifications: ", r_id_, notifications_[0], notifications_[1])
                        if notifications_[0] != notifications_[1]:

                                if str(task_dictionary["twilio_server"]["send_sms"]) == "true":
                                        account_sid = str(task_dictionary["twilio_server"]["account_sid"])
                                        auth_token = str(task_dictionary["twilio_server"]["auth_token"])
                                        from_number = str(task_dictionary["twilio_server"]["from_number"])
                                        to_number = str(task_dictionary["twilio_server"]["to_number"])
                                        client = Client(account_sid, auth_token)
                                        message = client.messages.create(
                                                        to = to_number,
                                                        from_ = from_number,
                                                        body = "Hello! Robot "+r_id_+" in fleet "+fleet_id+" requires attention "+notifications_[0]+". Please inspect. ")
                                        # print(message.sid, "--> account_sid: ",account_sid, "auth_token: ",auth_token, "from_number: ",from_number, "to_number: ",to_number)
                                        print("[manager]:-twilio sms auth_token verified.")

                                notifications_[1] = notifications_[0]
                                self.cur.execute('UPDATE table_robot SET notifications = %s WHERE robot_id = %s;', (','.join(notifications_), r_id_,))
                                self.conn.commit()
        except (ValueError, Exception, TypeError) as error:
                print("[manager]:-check_notification_msgs-"+str([error])+".")

# ---------------------------------------------- #
#      MAIN                                      #
# ---------------------------------------------- #

if __name__ == "__main__":
        Ui_MainWindow()
