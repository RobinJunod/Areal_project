# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np 
import matplotlib.pyplot as plt

# Landing and takeoff parameters
flying_height = 0.35
landed_height = 0.02
landing_heigh_steps = 0.002
liftoff_heigh_steps = 0.008
pad_centering_threshold = 0.05

# Speed parameters
crossing_speed = 0.2
exploration_speed = 0.3
nominal_return_speed = 0.3
landing_pad_scanning_speed = 0.2
P_vel = 1.5
vel_x_target, vel_y_target = 0, 0 # Global variable used for smoothing
speed_stable_threshold = 0.03

# Obstacle parameters
close_obstacle_limit = 0.2
edge_detection_threshold = 0.04 # For landing pad edges

# Yawing parameters
yawing_P = 0.5
yawing_D = 0.3
scanning_yaw_rate = 0.3

# Map parameters
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
landing_region_x_limit = 3.5 
landing_pad_size = 0.3
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.05 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

index_current_setpoint = 0
max_setpoints = max_y/res_pos * (max_x-landing_region_x_limit)/landing_pad_size -1


class MyController():
    def __init__(self):
        # States are : TAKEOFF, CROSSING_MAP, SEARCHING_LANDING_PAD, LANDING_PAD_CENTERING, GOAL_LANDING, RETURN_HOME, LANDING, LANDED
        self.state = 'TAKEOFF'

        self.height_desired = 0

        self.x_drone, self.y_drone = 0,0
        self.liftoff_point_x, self.liftoff_point_y = 0.5, 2.15
        self.goal_x, self.goal_y = 4.5,0.15 #Initial goal set to middle of end zone

        self.previous_range_down = 0

        self.stopped_on_pad_flag = False
        self.landing_pad_x_pos_found = False
        self.landing_pad_y_pos_found = False
        self.searching_x_flag = False

        self.landing_pad_x_pos = 0
        self.landing_pad_y_pos = 0

        self.x0 = 0
        self.y0 = 0
        self.yaw0 = 0

    def step_control(self, sensor_data):
        global index_current_setpoint, map

        self.x_drone = sensor_data['x_global'] - self.x0
        self.y_drone = sensor_data['y_global'] - self.y0
        self.yaw = sensor_data['yaw'] - self.yaw0

        map = occupancy_map(sensor_data, self.x_drone, self.y_drone, self.yaw)

        #print(self.x_drone, self.y_drone)

        ####### State machine #######

        #############################
        ######## MAIN STATES ########
        #############################

        # CROSSING_MAP STATE #
        if self.state == 'CROSSING_MAP':
            
            print(self.x_drone, self.y_drone)

            control_command = track_setpoint(self.goal_x, self.goal_y, self.x_drone, self.y_drone, sensor_data, self.height_desired, crossing_speed, self.yaw)

            if self.x_drone > landing_region_x_limit :
                control_command = [0, 0, 0, self.height_desired]
                
                self.state = 'SEARCHING_LANDING_PAD'

            self.previous_range_down = sensor_data['range_down']

            return control_command

        # SEARCHING_LANDING_PAD #
        elif self.state == 'SEARCHING_LANDING_PAD':
            
            # If vertical edge detected -> change state to LANDING_PAD_CENTERING
            if abs(sensor_data['range_down'] - self.previous_range_down) > edge_detection_threshold :
                self.state = 'LANDING_PAD_CENTERING_Y'
                control_command = [0.0, 0.0, 0.0, self.height_desired]
                if index_current_setpoint % (int(max_y/res_pos)*2) > int(max_y/res_pos) :
                    self.landing_pad_y_pos = self.y_drone - landing_pad_size/2
                else :
                    self.landing_pad_y_pos = self.y_drone + landing_pad_size/2

                self.landing_pad_y_pos_found = True
                self.landing_pad_x_pos = self.x_drone

                return control_command
            
            self.previous_range_down = sensor_data['range_down']

            # Get the goal position and drone position
            x_target, y_target = find_exploration_setpoint()
            distance_drone_to_goal = np.linalg.norm([x_target - self.x_drone, y_target - self.y_drone])

            # When the drone reaches the goal setpoint
            if distance_drone_to_goal < 0.1:
                # Select the next setpoint as the goal position
                index_current_setpoint += 1

                if index_current_setpoint > max_setpoints :
                    index_current_setpoint = 0

            # Calculate the control command based on current goal setpoint
            x_target, y_target = find_exploration_setpoint()

            control_command = track_setpoint(x_target, y_target, self.x_drone, self.y_drone, sensor_data, self.height_desired, exploration_speed, self.yaw)

            return control_command

        # LANDING_PAD_CENTERING Y STATE #
        elif self.state == 'LANDING_PAD_CENTERING_Y':

            control_command = [0.0, 0.0, 0.0, self.height_desired]

            goal_dist = compute_dist(self.landing_pad_x_pos, self.landing_pad_y_pos, self.x_drone, self.y_drone)

            centering_speed = goal_dist/(landing_pad_size*2) # *2 for less agressive movement
            centering_speed = np.clip(centering_speed, 0, nominal_return_speed)

            control_command = track_setpoint(self.landing_pad_x_pos, self.landing_pad_y_pos, self.x_drone, self.y_drone, sensor_data, self.height_desired, centering_speed, self.yaw)

            print("LANDING_PAD_CENTERING_Y")

            # If well centered, change state to land on the goal
            if goal_dist < pad_centering_threshold and abs(sensor_data['v_forward']) < speed_stable_threshold and abs(sensor_data['v_left']) < speed_stable_threshold:
                self.state = 'LANDING_PAD_CENTERING_X'

            return control_command

        # LANDING_PAD_CENTERING STATE #
        elif self.state == 'LANDING_PAD_CENTERING_X':

            control_command = [0.0, 0.0, 0.0, self.height_desired]

            if abs(sensor_data['range_down'] - self.previous_range_down) > edge_detection_threshold :
                self.landing_pad_x_pos = self.x_drone - landing_pad_size/2
                self.state = 'LANDING_PAD_CENTERING_XY'
                return control_command

            goal_dist = compute_dist(self.landing_pad_x_pos + landing_pad_size + 0.15, self.landing_pad_y_pos, self.x_drone, self.y_drone)

            centering_speed = goal_dist/(landing_pad_size*2) # *2 for less agressive movement
            centering_speed = np.clip(centering_speed, 0, nominal_return_speed)

            control_command = track_setpoint(self.landing_pad_x_pos + landing_pad_size  + 0.15, self.landing_pad_y_pos, self.x_drone, self.y_drone, sensor_data, self.height_desired, centering_speed, self.yaw)
            
            print("LANDING_PAD_CENTERING_X")
            print(goal_dist)

            # If well centered, change state to land on the goal
            if goal_dist < pad_centering_threshold and abs(sensor_data['v_forward']) < speed_stable_threshold and abs(sensor_data['v_left']) < speed_stable_threshold:
                self.landing_pad_x_pos = self.landing_pad_x_pos - 0.6

            return control_command
        
                # LANDING_PAD_CENTERING STATE #
        elif self.state == 'LANDING_PAD_CENTERING_XY':

            control_command = [0.0, 0.0, 0.0, self.height_desired]

            goal_dist = compute_dist(self.landing_pad_x_pos, self.landing_pad_y_pos, self.x_drone, self.y_drone)

            centering_speed = goal_dist/(landing_pad_size*2) # *2 for less agressive movement
            centering_speed = np.clip(centering_speed, 0, nominal_return_speed)

            control_command = track_setpoint(self.landing_pad_x_pos, self.landing_pad_y_pos, self.x_drone, self.y_drone, sensor_data, self.height_desired, centering_speed, self.yaw)

            print("LANDING_PAD_CENTERING_XY")

            # If well centered, change state to land on the goal
            if goal_dist < pad_centering_threshold and abs(sensor_data['v_forward']) < speed_stable_threshold and abs(sensor_data['v_left']) < speed_stable_threshold:
                self.state = 'GOAL_LANDING'

            return control_command
        
        # RETURN_HOME STATE #
        elif self.state == 'RETURN_HOME':

            print(self.x_drone, self.y_drone)

            if self.height_desired < flying_height :
                self.height_desired += liftoff_heigh_steps
                control_command = [0, 0, 0, self.height_desired]
            else :
                goal_dist = compute_dist(self.goal_x, self.goal_y, self.x_drone, self.y_drone)

                return_speed = goal_dist/(landing_pad_size*2)
                return_speed = np.clip(return_speed, 0, nominal_return_speed)

                control_command = track_setpoint(self.goal_x, self.goal_y, self.x_drone, self.y_drone, sensor_data, self.height_desired, return_speed, self.yaw)

                if goal_dist < pad_centering_threshold and abs(sensor_data['v_forward']) < speed_stable_threshold and abs(sensor_data['v_left']) < speed_stable_threshold:
                    control_command = [0, 0, 0, self.height_desired]
                    self.state = 'LANDING'

            return control_command
    
        ##########################
        #### SECONDARY STATES ####
        ##########################
        # LANDED STATE #
        elif self.state == 'LANDED': 
            control_command = [0.0, 0.0, 0.0, 0.0]
            return control_command

        # TAKEOFF STATE #
        elif self.state == 'TAKEOFF':
            print(self.x_drone, self.y_drone)

            if self.height_desired == 0 :
                self.yaw0 = sensor_data['yaw']
                self.x0 = sensor_data['x_global'] - self.liftoff_point_x
                self.y0 = sensor_data['y_global'] - self.liftoff_point_y

        
            if self.height_desired < flying_height :
                self.height_desired += liftoff_heigh_steps

            control_command = [0.0, 0.0, scanning_yaw_rate, self.height_desired]

            if flying_height < self.height_desired :
                self.state = 'CROSSING_MAP'

            return control_command

        # LANDING STATE #
        elif self.state == 'LANDING':
            self.height_desired -= landing_heigh_steps
            control_command = [0.0, 0.0, 0.0, self.height_desired]

            if sensor_data['range_down'] < landed_height :
                self.state = 'LANDED'
                control_command = [0.0, 0.0, 0.0, 0.0]

            return control_command
        
        # GOAL_LANDING STATE #
        elif self.state == 'GOAL_LANDING':
            print("Landing on goal")
            self.height_desired -= landing_heigh_steps
            control_command = [0.0, 0.0, 0.0, self.height_desired]

            if sensor_data['range_down'] < landed_height :
                self.state = 'RETURN_HOME'
                self.goal_x, self.goal_y = self.liftoff_point_x, self.liftoff_point_y

            return control_command

def track_setpoint(setpoint_x, setpoint_y, x_drone, y_drone, sensor_data, height_desired, nominal_speed, yaw) :
    global vel_x_target, vel_y_target

    # Constant yaw_rate to enhance map generation
    yaw_rate_target = scanning_yaw_rate

    # Calculate forces with potential field algorithm
    force_x, force_y = potential_field(x_drone, y_drone, setpoint_x, setpoint_y)

    # Calculate the velocity targets in drone reference frame
    new_vel_x_target = nominal_speed * (force_x * np.cos(yaw) + force_y * np.sin(yaw))
    new_vel_y_target = nominal_speed * (- force_x * np.sin(yaw) + force_y * np.cos(yaw))

    new_vel_x_target = np.clip(new_vel_x_target, -nominal_speed, nominal_speed)
    new_vel_y_target = np.clip(new_vel_y_target, -nominal_speed, nominal_speed)

    # Close obstacles avoidance
    if sensor_data['range_front'] < close_obstacle_limit :
        new_vel_x_target = np.clip(new_vel_x_target, -nominal_speed, 0)
    if sensor_data['range_back'] < close_obstacle_limit :
        new_vel_x_target = np.clip(new_vel_x_target, 0, nominal_speed)
    if sensor_data['range_right'] < close_obstacle_limit :
        new_vel_y_target = np.clip(new_vel_y_target, 0, nominal_speed)
    if sensor_data['range_left'] < close_obstacle_limit :
        new_vel_y_target = np.clip(new_vel_y_target, -nominal_speed, 0)

    # Control signal smoothing to avoid unstabilities in the lower level controller
    vel_x_target = 0.9 * vel_x_target + 0.1 * new_vel_x_target
    vel_y_target = 0.9 * vel_y_target + 0.1 * new_vel_y_target

    control_command = [vel_x_target, vel_y_target, yaw_rate_target, height_desired]

    return control_command

def potential_field(pos_x, pos_y, goal_x, goal_y):
    obs_radius = 0.5 #Obstacles influence radius
    k_rep = 0.1 # Obstacles repulsion force

    # Calculate attractive force
    f_att = np.array([goal_x - pos_x, goal_y - pos_y])
    f_att /= np.linalg.norm(f_att)

    # Calculate repulsive force
    f_rep = np.zeros(2)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i,j] < 0 :
                # Calculate distance between current position and map cell
                cell_pos_x = i * res_pos + res_pos/2
                cell_pos_y = j * res_pos + res_pos/2
                dist = compute_dist(pos_x,pos_y,cell_pos_x,cell_pos_y)

                # Calculate repulsive force if within obstacle radius
                if dist < obs_radius:
                    f_rep[0] += k_rep * ((1/dist) - (1/obs_radius)) * ((pos_x - cell_pos_x)/dist)
                    f_rep[1] += k_rep * ((1/dist) - (1/obs_radius)) * ((pos_y - cell_pos_y)/dist)

    # Combine attractive and repulsive forces
    f_total = f_att + f_rep
    f_total /= np.linalg.norm(f_total)

    f_total = np.clip(f_total, -1, 1) # Forces can never be more than 1

    return f_total[0], f_total[1]

def find_exploration_setpoint():
    global index_current_setpoint
    x_setpoint, y_setpoint = 0, 0
    i, j = 0, 0

    # Compute the current target setpoint to explore the landing region
    i, j = compute_ij_setpoint()

    # Skip the setpoint if there is a nearby obstacle
    skip_condition = isObstacle(i,j)

    while skip_condition :
        index_current_setpoint = index_current_setpoint+1

        # Loop exploration for the case were the map wasn't totally explored (increases research reliability)
        if index_current_setpoint > max_setpoints :
            index_current_setpoint = 0

        i, j = compute_ij_setpoint()
        skip_condition = isObstacle(i,j)

    # Compute real xy position target, i compensated to correspond to the map x index
    x_setpoint = landing_region_x_limit + landing_pad_size/2 + i * landing_pad_size
    y_setpoint = res_pos/2 + j * res_pos

    return x_setpoint, y_setpoint

def compute_ij_setpoint():
    # Creates Zig-Zag pattern in the landing pad research zone
    # Warning! It does not match the x axis map indexes

    i = int(index_current_setpoint / (max_y/res_pos))
    
    if (i % 2) == 0 :
        j = index_current_setpoint%int(max_y/res_pos)
    else :
        j = (max_y/res_pos) - index_current_setpoint%int(max_y/res_pos) - 1

    return int(i), int(j)

def isObstacle(i, j):
    # The goal of this function is to find is a setpoint is an obstacle or near an obstacle
    proximity_threshold = 0.2

    # i index compensation to correspond to the map x index
    map_i = int(i * landing_pad_size/res_pos + int(landing_region_x_limit/res_pos) + landing_pad_size/2/res_pos)

    for x in range(max(0, map_i-int(proximity_threshold/res_pos)), min(map_i+int(proximity_threshold/res_pos), map.shape[0])):
        for y in range(max(0, j-int(proximity_threshold/res_pos)), min(j+int(proximity_threshold/res_pos), map.shape[1])):
            if map[x,y] <= 0.0: # If the certainty of the map in this area is high enough, assume it's an obstacle, <= to zero to avoid having setpoints inside big obstacles with map[x,y] = 0
                distance = np.sqrt((map_i-x)**2 + (j-y)**2) * res_pos # Calculate the distance in meters
                
                if (distance <= proximity_threshold):
                    return True # The setpoint is too close to an obstacle or is an obstacle
                
    return False # The setpoint is safe to explore

def compute_dist(point1_x, point1_y, point2_x, point2_y):
    # Compute distance from point 1 to point 2
    x_dist = point2_x - point1_x
    y_dist = point2_y - point1_y

    dist = np.linalg.norm([x_dist, y_dist])

    return dist

### OCCUPANCY MAP FUNCTION FROM EXAMPLES ###
def occupancy_map(sensor_data, pos_x, pos_y, yaw):
    global map, t
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the point is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break

    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
    t +=1

    return map