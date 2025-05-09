# ====== =====    ====   ====  ====  = ===== =====
#     =      =    =   = =    = =   = =   =   =
#    =     ===    ====  =    = ====  =   =   =====
#   =        =    =  =  =    = =   = =   =       =
#  =     =====    =   =  ====  ====  =   =   =====
# ======================================================================================

from controller import Robot, Emitter
import math
import numpy as np 
import cv2 
import struct
import matplotlib.pyplot as plt
import os

# ==================== INITIALIZATION ====================

# Define o timeStep e a velocidade máxima
timeStep = 16
max_velocity = 6.28 
timeStep2 = timeStep * 20
robot_radius = 0.037

# Cria instância do robô
robot = Robot()

# Definição de rodas e sensores de navegação
wheel_L = robot.getDevice("wheel2 motor")  
wheel_R = robot.getDevice("wheel1 motor")  
wheel_L.setPosition(float("inf"))
wheel_R.setPosition(float("inf"))

gps = robot.getDevice("gps")
gyro = robot.getDevice("gyro")
lidar = robot.getDevice("lidar")

# Definição de sensores de reconhecimento de vítimas
colour = robot.getDevice("colour_sensor")
camera_R = robot.getDevice("camera2") # Câmera direita
camera_L = robot.getDevice("camera1") # Câmera esquerda

# Definição do Emitter
emitter = robot.getDevice("emitter")

# Liga todos os sensores 
gps.enable(timeStep)
gyro.enable(timeStep)
lidar.enable(timeStep2)
lidar.enablePointCloud()     
colour.enable(timeStep2)
camera_R.enable(timeStep)
camera_L.enable(timeStep)

# ==================== VARIABLES ====================

# Ticks
tick = 0
tick_action = 0
tick_areas = 0
tick_buffer = 0
tick_continue = 0
tick_swamp = 0
tick_checkpoint = 0
tick_holes = 0

# GPS
values_gps = [0.0, 0.0, 0.0]
initial_position = [0.0, 0.0]
initial_node = [0.0, 0.0]
current_position = [0.0, 0.0]
current_node = [0.0, 0.0]
old_node = [0.0, 0.0]
position_flag = False
target_position = [0.0, 0.0]

# GYRO
values_gyro = [0.0, 0.0, 0.0]
initial_angle = 0.0
current_angle = 0.0

# LIDAR
values_lidar = [0.0] * 512
values_pointCloud = [] * 512
camera_L_lidar = [0.0] * 170
camera_R_lidar = [0.0] * 170

# Color
colour_data = [0.0, 0.0, 0.0]

# Action
action_type = "none"
action_parameter = "none"

# Token
collected_tokens = {'S': [], 'U': [], 'H': [], 'O': [], 'F': [], 'C': [], 'P': []}

# Walls
walls = {}

# Graph
graph_map = {}
borders = []
margin = []
interior = []
holes = []
swamps = []
checkpoints = []

node_length = 0.04
tile_length = 0.12

# Flags
hole_flag = False 
continue_send_token_flag = False
current_mood = "exploration_mood"
send_map_flag = False

# Areas
current_area = 1
transition_areas = []

# Checkpoints
last_checkpoint = [0,0]

# Empty map 43 x 43 with '0' char values
max_tile = 41 * 2 + 1
initial_tile = 41
compact_map = np.full((max_tile, max_tile), '0', dtype=str)
aux_map = np.full((max_tile, max_tile), '0', dtype=str)
full_map = np.full((2*max_tile-1, 2*max_tile-1), '0', dtype=str)

# ==================== FUNCTIONS ====================

   # print("######################")
   # print("Tick: ", tick)
   # print("     Posicao = ", current_position, "Posicao inicial = ", initial_position)
   # print("     Angulo = ", current_angle)
   # print("     Action = ", action_type, action_parameter)
   # print("         Lidar = ", np.max(values_lidar[0:511]), len(values_lidar))
   # print("         len(graph_map) = ", len(graph_map), " len(borders) = ", len(borders), " len(interior) = ", len(interior))
   # print("         current_node = ", current_node)
   # print("         collected_tokens = ", collected_tokens)
   # print("         len(values_pointCloud) = ", len(values_pointCloud))
   # print("     len(walls) = ", len(walls))
   # print("     Colour data = ", colour_data)

# Function to save the plot for each tick
def save_plot():
    global walls, initial_position, current_position, tick

    # Create the plot
    plt.figure(figsize=(6, 6))


    # For each node in the graph, plot the node as a square color in the position
    for node in graph_map:
        x, y = node

        #Calculate the position from the node
        x_pos, y_pos = node2position(x, y)

        # Plot the node as an orange square if interior node and green if border node and grey otherwise
        if node in margin:
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color='purple', alpha=0.25))
        elif node in borders: 
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color='green', alpha=0.25))
        elif node in interior:
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(207/255, 207/255, 207/255), alpha=1))
        else:
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color='grey', alpha=0.25))

    # Plot the walls (connections between nodes)
    for node, neighbors in walls.items():
        x, y = node
        for neighbor in neighbors:
            x1, y1 = neighbor
            plt.plot([x, x1], [y, y1], color = (85/255,170/255,183/255), linewidth=5, alpha=1)


    for node in holes:
        x, y = node

        #Calculate the position from the node
        x_pos, y_pos = node2position(x, y)
        plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(10/255, 10/255,10/255),  alpha=1))
        
    for node in swamps:
        x, y = node

        #Calculate the position from the node
        x_pos, y_pos = node2position(x, y)
        plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(182/255, 148/255, 83/255), alpha=1))

    for node in checkpoints:
        x, y = node

        #Calculate the position from the node
        x_pos, y_pos = node2position(x, y)
        plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(59/255, 63/255, 78/255), alpha=1))        
        
    for node in transition_areas:
        x,y = node[0]
        transition_type = node[1]

        #Calculate the position from the node
        x_pos, y_pos = node2position(x, y)

        if transition_type == 'b':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(52/255, 52/255, 242/255), alpha=1))
        elif transition_type == 'y':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(242/255, 242/255, 52/255), alpha=1))
        elif transition_type == 'g':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(27/255, 236/255, 27/255), alpha=1))
        elif transition_type == 'p':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(120/255, 52/255, 201/255), alpha=1))
        elif transition_type == 'o':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(242/255, 201/255, 52/255), alpha=1))
        elif transition_type == 'r':
            plt.gca().add_patch(plt.Rectangle((x_pos - node_length / 2, y_pos - node_length / 2), node_length, node_length, color=(242/255, 52/255, 52/255), alpha=1))
    # For each edge in the graph, plot the edge as a black line if the distance is not infinite
    for node, neighbors in graph_map.items():
        x, y = node
        x, y = node2position(x, y)
        for neighbor, weight in neighbors.items():
            if weight < 1e5:
                x1, y1 = neighbor
                x1, y1 = node2position(x1, y1)
                plt.plot([x, x1], [y, y1], 'k-', alpha=0.1, linewidth=1)
    
    for token in collected_tokens:
        for token_position in collected_tokens[token]:
            x, y = token_position

            # Add the letter itself to the plot 
            plt.text(x, y, token, fontsize=12, ha='center', va='center', color='black')
        
    # Plot the current position in blue
    plt.plot(current_position[0], current_position[1], 'go', label='Current Position')

    # Plot limit from -1, -1 to 1, 1
    plt.xlim(-0.12, 1.08)
    plt.ylim(-1.08, 0.12)
  
    # plt.xlim(-1.25, 0.25)
    # plt.ylim(-0.5, 1)

    # Label the axes
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Add a title
    plt.title(f'Walls and Positions - Tick {tick}')

    # Show the grid for better visualization
    plt.grid(True)

    # Save the plot to a file
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/plot_tick_{tick}.png")

    # Clear the plot for the next tick
    plt.close()

def update_walls(step = 16):
    #print("              Updating walls")

    global walls, values_pointCloud, current_position, current_angle, initial_angle

    # Iterate over the point cloud to create walls
    for i in range(0, len(values_pointCloud), step):
        # Get current and next wall points (wrap around at the end)
        current_point = (values_pointCloud[i].x, values_pointCloud[i].y)
        next_point = (values_pointCloud[(i + step) % len(values_pointCloud)].x, 
                      values_pointCloud[(i + step) % len(values_pointCloud)].y)
        #print("              Peguei um ponto")
        #print("                 current_angle = ", current_angle, " initial_angle = ", initial_angle)
        #print("                 current_position = ", current_position)
        #print("                   current_point = ", current_point, " next_point = ", next_point)
        
        delta_angle =  current_angle
        current_point = (np.cos(delta_angle) * (-current_point[0]) - np.sin(delta_angle) * (-current_point[1]),
                         np.sin(delta_angle) * (-current_point[0]) + np.cos(delta_angle) * (-current_point[1]))
        next_point = (np.cos(delta_angle) * (-next_point[0]) - np.sin(delta_angle) * (-next_point[1]),
                        np.sin(delta_angle) * (-next_point[0]) + np.cos(delta_angle) * (-next_point[1]))
        #print("                   current_point = ", current_point, " next_point = ", next_point)        

        if (distance((0,0),current_point) > node_length*1.5*np.sqrt(2)) or (distance((0,0),next_point) > node_length*1.5*np.sqrt(2)):
            continue

        initial_wall = position2wall(current_point[0] + current_position[0], current_point[1]  + current_position[1])
        final_wall = position2wall(next_point[0] + current_position[0], next_point[1] + current_position[1])
        #print("              Tentando conectar paredes" + str(initial_wall) + str(final_wall))

        # If walls are close enough, connect them
        if 0 < distance(initial_wall, final_wall) <= 0.025 * step/8:
            #print("                 Conectando paredes")

            if initial_wall not in walls:
                walls[initial_wall] = []
            if final_wall not in walls:
                walls[final_wall] = []
            
            if initial_wall not in walls[final_wall]:
                add_walls(initial_wall, final_wall)


def update_holes():
    global holes, tick_action, hole_flag, tick_holes

    if tick_holes > 0:
        return
    
    if colour_data[0] < 40 and colour_data[1] < 40 and colour_data[2] < 40 or colour_data[0] == 107 and colour_data[1] == 107 and colour_data[2] == 107:
       # print("                 Hole detected")
       # print("                 Colour data = ", colour_data)
        
        hole_position = [current_position[0] + (robot_radius+0.02) * np.cos(current_angle), current_position[1] + (robot_radius+0.02) * np.sin(current_angle)]
        hole_tile = position2tile(hole_position[0], hole_position[1], type = "tile")
        hole_node = tile2node(hole_tile[0], hole_tile[1], type = "tile")
        
        # Check if a neighboring area already exists
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if np.abs(i) + np.abs(j) == 1:
                    if ((hole_node[0] + 3*i, hole_node[1] + 3*j)) in holes:
                        #print("                 Holes já existe")
                        return
        
        # Add the 3x3 tile to the holes list
        for i in range(-2,3,1):
            for j in range(-2,3,1):
                # print("                 i = ", i, " j = ", j)
                target_node = (hole_node[0] + i, hole_node[1] + j)
                add_node((target_node[0], target_node[1]), type = "margin")
                if target_node not in holes and np.abs(i) <= 1 and np.abs(j) <= 1:
                    holes.append(target_node)
                
        tick_action = 0
        hole_flag = True
        tick_holes = 100
        if current_area != 4:
          update_map("holes", hole_tile, "none")
        else:
            update_map("area4", hole_tile, "none")

def update_swamps():
    global swamps, tick_swamp

    if tick_swamp > 0 or (np.abs(current_angle - np.round(current_angle/(np.pi/2)) * np.pi/2) > np.pi/16):
        return 
    
    R = colour_data[0]
    G = colour_data[1] 
    B = colour_data[2]

    if R > 170 and R < 190 and G > 140 and G < 160 and B > 70 and B < 90:
        swamp_position = [current_position[0] + (robot_radius+0.02) * np.cos(current_angle), current_position[1] + (robot_radius+0.02) * np.sin(current_angle)]
        swamp_tile = position2tile(swamp_position[0], swamp_position[1], type = "tile")
        swamp_node = tile2node(swamp_tile[0], swamp_tile[1], type = "tile")

        # Check if a neighboring area already exists
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if np.abs(i) + np.abs(j) == 1:
                    if ((swamp_node[0] + 3*i, swamp_node[1] + 3*j)) in swamps:
                        #print("                 Swamp já existe")
                        return
        
        # Add the 3x3 tile to the swamps list
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                target_node = (swamp_node[0] + i, swamp_node[1] + j)
                if target_node not in swamps:
                    swamps.append(target_node)
        
        tick_swamp = 200
        if current_area != 4:
          update_map("swamp", swamp_tile, "none")
        else:
          update_map("area4", swamp_tile, "none")
                
        
def update_areas():
    global current_area, transition_areas, tick_areas

    if current_area == 4 and tick_areas == 0:
        update_map("area4", position2tile(current_position[0], current_position[1], type = "tile"), "none")
    
    if tick_areas > 0 or (np.abs(current_angle - np.round(current_angle/(np.pi/2)) * np.pi/2) > np.pi/16):
        return 

    R = colour_data[0]
    G = colour_data[1]
    B = colour_data[2]
        
    areas_position = [current_position[0] + (robot_radius+0.02) * np.cos(current_angle), current_position[1] + (robot_radius+0.02) * np.sin(current_angle)]
    areas_tile = position2tile(areas_position[0], areas_position[1], type = "tile")
    areas_node = tile2node(areas_tile[0], areas_tile[1], type = "tile")
    
    transition_type = False
    
    if R > 40 and R < 60 and G > 40 and G < 60 and B > 230 and B < 250:
       # print("                 Nova area: 2, Area antigo: 1")

        if(current_area == 1):
            current_area = 2
        elif(current_area == 2):
            current_area = 1

        transition_type = 'b'

    elif R > 230 and R < 250 and G > 230 and G < 250 and B > 40 and B < 60:
        # print("                 Nova area: 3, Area antigo: 1")
        
        if(current_area == 1):
            current_area = 3
        elif(current_area == 3):
            current_area = 1
            
        transition_type = 'y'
        
    elif R > 20 and R < 40 and G > 230 and G < 250 and B > 20 and B < 40:
        #print("                 Nova area: 4, Area antigo: 1")
        
        if(current_area == 1):
            current_area = 4
        elif(current_area == 4):
            current_area = 1
        
        transition_type = 'g'

    elif R > 110 and R < 130 and G > 40 and G < 60 and B > 190 and B < 210:
        #print("                 Nova area: 3, Area antigo: 2")
        
        if(current_area == 2):
            current_area = 3
        elif(current_area == 3):
            current_area = 2
            
        transition_type = 'p'

    elif R > 230 and R < 250 and G > 190 and G < 210 and B > 40 and B < 60:
        #print("                 Nova area: 4, Area antigo: 2")
        
        if(current_area == 2):
            current_area = 4
        elif(current_area == 4):
            current_area = 2
        
        transition_type = 'o'
        
    elif R > 230 and R < 250 and G > 40 and G < 60 and B > 40 and B < 60:
        #print("                 Nova area: 4, Area antigo: 3")

        if(current_area == 3):
            current_area = 4
        elif(current_area == 4):
            current_area = 3
    
        transition_type = 'r'
    
    if transition_type != False:
        # Check if a neighboring area already exists
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if np.abs(i) + np.abs(j) == 1:
                    if [(areas_node[0] + 3*i, areas_node[1] + 3*j), transition_type] in transition_areas:
                        #print("                 Area já existe")
                        return
                    
        # Add the 3x3 tile to the transition_areas list
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                target_node = (areas_node[0] + i, areas_node[1] + j)
                if target_node not in transition_areas:
                    transition_areas.append([target_node, transition_type])
        
        update_map("transition_areas", areas_tile, transition_type)

        tick_areas = 500

def update_checkpoints():
    global checkpoints, last_checkpoint, tick_checkpoint

    if tick_checkpoint > 0 or (np.abs(current_angle - np.round(current_angle/(np.pi/2)) * np.pi/2) > np.pi/16):
        return 

    R = colour_data[0]
    G = colour_data[1]
    B = colour_data[2]

    if is_checkpoint_color(R, G, B) == True:
        #print("                 Checkpoint detected")
        #print("                 Colour data = ", colour_data)

        checkpoint_position = [current_position[0] + (robot_radius+0.02) * np.cos(current_angle), current_position[1] + (robot_radius+0.02) * np.sin(current_angle)]
        checkpoint_tile = position2tile(checkpoint_position[0], checkpoint_position[1], type = "tile")
        checkpoint_node = tile2node(checkpoint_tile[0], checkpoint_tile[1], type = "tile")
        
        # Check if a neighboring area already exists
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if np.abs(i) + np.abs(j) == 1:
                    if ([checkpoint_node[0] + i, checkpoint_node[1] + j]) in checkpoints:
                        #print("                 Checkpoint já existe")
                        return
        
        # Add the 3x3 tile to the checkpoints list
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                target_node = (checkpoint_node[0] + i, checkpoint_node[1] + j)
                if target_node not in checkpoints:
                    checkpoints.append(target_node)
        
        if current_area != 4:
          update_map("checkpoints", checkpoint_tile, "none")
        else:
          update_map("area4", checkpoint_tile, "none")
        
            

        tick_checkpoint = 500

def is_checkpoint_color(R, G, B):

    # Agora checar se está em alguma faixa válida de checkpoint
    checkpoint_ranges = [
        ((64, 74), (68, 78), (80, 90)),
        ((33, 43), (40, 50), (59, 69)),
        ((57, 67), (60, 70), (72, 82)),
        ((51, 61), (56, 66), (69, 79)),
        ((56, 66), (60, 70), (73, 83)),
        ((50, 60), (54, 64), (68, 78)),
        ((57, 67), (61, 71), (74, 84)),
        ((63, 73), (66, 76), (76, 86)),
        ((51, 61), (55, 65), (70, 80)),
        ((52, 62), (56, 66), (70, 80)),
        ((53, 63), (57, 67), (70, 80)),
        ((55, 65), (58, 68), (70, 80)),
        ((50, 60), (54, 64), (67, 77)),
        ((59, 69), (62, 72), (72, 82)),
        ((54, 64), (57, 67), (70, 80)),
        ((59, 69), (62, 72), (74, 84)),
        ((51, 61), (55, 65), (69, 79)),
        ((53, 63), (56, 66), (68, 78)),
        ((55, 65), (58, 68), (70, 80)),
        ((55, 65), (58, 68), (70, 80)),
        ((55, 65), (58, 68), (70, 80)),
        ((52, 62), (56, 66), (69, 79)),
        ((28, 38), (34, 44), (54, 64)),
        ((30, 40), (37, 47), (56, 66)),
        ((31, 41), (37, 47), (57, 67)),
        ((31, 41), (35, 45), (54, 64)),
        ((30, 40), (37, 47), (56, 66)),
        ((29, 39), (36, 46), (55, 65)),
        ((27, 37), (34, 44), (54, 64)),
        ((30, 40), (36, 46), (55, 65)),
        ((29, 39), (35, 45), (56, 66)),
        ((27, 37), (34, 44), (55, 65)),
        ((29, 39), (35, 45), (55, 65)),
        ((28, 38), (34, 44), (53, 63)),
        ((27, 37), (34, 44), (54, 64)),
        ((29, 39), (36, 46), (56, 66)),
        ((28, 38), (35, 45), (56, 66)),
        ((29, 39), (36, 46), (55, 65)),
        ((27, 37), (34, 44), (54, 64)),
        ((28, 38), (34, 44), (53, 63)),
        ((28, 38), (35, 45), (56, 66)),
        ((26, 36), (33, 43), (52, 62)),
        ((75, 85), (78, 88), (89, 99)),
        ((93, 103), (96, 106), (104, 114)),
        ((77, 87), (80, 90), (90, 100)),
        ((73, 83), (76, 86), (86, 96)),
        ((87, 97), (91, 101), (99, 109)),
        ((78, 88), (78, 88), (91, 101)),
        ((73, 83), (76, 86), (86, 96)),
        ((91, 101), (94, 104), (102, 112)),
        ((77, 87), (80, 90), (90, 100)),
        ((85, 95), (87, 97), (96, 106)),
        ((84, 94), (87, 97), (95, 105)),
        ((94, 104), (96, 106), (103, 113)),
        ((78, 88), (80, 90), (90, 100)),
        ((89, 99), (91, 101), (98, 108)),
        ((83, 93), (86, 96), (96, 106)),
        ((94, 104), (96, 106), (103, 113)),
        ((80, 90), (83, 93), (91, 101)),
        ((94, 104), (96, 106), (103, 113)),
        ((82, 92), (85, 95), (93, 103)),
        ((95, 105), (97, 107), (106, 116)),
        ((35, 45), (42, 52), (62, 72)),
        ((37, 47), (44, 54), (63, 73)),
        ((40, 50), (47, 57), (67, 77)),
        ((35, 45), (41, 51), (61, 71)),
        ((35, 45), (42, 52), (62, 72)),
        ((39, 49), (46, 56), (66, 76)),
        ((35, 45), (42, 52), (62, 72)),
        ((38, 48), (45, 55), (64, 74)),
        ((35, 45), (42, 52), (61, 71)),
        ((34, 44), (41, 51), (62, 72)),
        ((35, 45), (42, 52), (63, 73)),
        ((37, 47), (44, 54), (62, 72)),
        ((36, 46), (42, 52), (61, 71)),
        ((34, 44), (41, 51), (61, 71)),
        ((36, 46), (43, 53), (63, 73)),
        ((35, 45), (42, 52), (62, 72)),
        ((39, 49), (46, 56), (67, 77)),
        ((35, 45), (42, 52), (62, 72)),
        ((40, 50), (47, 57), (67, 77)),
        ((38, 48), (45, 55), (63, 73)),
    ]


    for (min_R, max_R), (min_G, max_G), (min_B, max_B) in checkpoint_ranges:
        if (min_R <= R <= max_R) and (min_G <= G <= max_G) and (min_B <= B <= max_B):
            return True  # Está em uma faixa reconhecida de checkpoint

    return False  # Fora das faixas conhecidas e fora das faixas de checkpoint

def add_walls(initial_wall, final_wall):
    global walls
    global target_position

    # If we are in areas 1 or 2, we only add horizontal and vertical walls
    if current_area == 1 or current_area == 2 or tick_areas > 0:
        if initial_wall[0] != final_wall[0] and initial_wall[1] != final_wall[1]:
            return

    walls[initial_wall].append(final_wall)
    walls[final_wall].append(initial_wall)   
    
    for wall in [initial_wall, final_wall]:
        
        wall_node = position2node(wall[0], wall[1])

        wall_tile = position2tile(wall[0], wall[1], type = "transition")
        wall_tile = position2tile(wall[0], wall[1], type = "transition")

        if current_area != 4:
            update_map("walls", wall_tile, "none")
        else:
            update_map("area4", wall_tile, "none")
    
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                target_node = position2node(wall[0] + i * node_length/2, wall[1] + j * node_length/2)
                
                add_node(target_node, type = "margin")


def first_ticks():
    global initial_position, initial_angle, current_angle, initial_node

    # Get initial position
    if tick == 0:
        initial_position = current_position
        initial_tile = position2tile(initial_position[0], initial_position[1], type = "tile")
        initial_node = position2node(initial_position[0], initial_position[1])
        update_map("initial_position", initial_tile, "none")
        
    # Get initial orientation
    if tick == 7:
        initial_angle = np.arctan2(current_position[1], current_position[0])
        if initial_angle < 0:
            initial_angle = initial_angle + 2 * np.pi
        if initial_angle > 2 * np.pi:
            initial_angle = initial_angle - 2 * np.pi
        current_angle = initial_angle
    

def update_sensors():
    global values_gps, current_position, current_node
    global values_gyro, current_angle
    global values_lidar, values_pointCloud
    global camera_L_data, camera_L_width, camera_L_height, camera_L_lidar
    global camera_R_data, camera_R_width, camera_R_height, camera_R_lidar
    global colour_data

    # ----- Navegation -----

    # Update GPS sensors    
    values_gps = gps.getValues()
    current_position = [values_gps[0], -values_gps[2]] 
    if(tick > 0):
        current_position[0] = current_position[0] - initial_position[0]
        current_position[1] = current_position[1] - initial_position[1]
    current_node = position2node(current_position[0], current_position[1])

    # Update GYRO sensors
    values_gyro = gyro.getValues()
    current_angle = current_angle + values_gyro[1] * timeStep / 1000
    if current_angle > 2 * np.pi:
        current_angle = current_angle - 2 * np.pi
    if current_angle < 0:
        current_angle = current_angle + 2 * np.pi
          
    # Update LIDAR sensors 
    values_lidar = lidar.getRangeImage()
    for i in range (2048):
        if str(values_lidar[i]) == "inf":
            values_lidar[i] = 1
    values_lidar = values_lidar[1024:1536]
    values_pointCloud = lidar.getPointCloud()
    values_pointCloud = values_pointCloud[1024:1536]

    
    # ----- Camera -----

    # Update Camera sensor
    camera_L_data = camera_L.getImage()
    camera_L_width = camera_L.getWidth()
    camera_L_height = camera_L.getHeight()
    camera_L_lidar = values_lidar[352 : 416]    # values_lidar[384:511] + values_lidar[0:41]

    camera_R_data = camera_R.getImage()
    camera_R_width = camera_R.getWidth()
    camera_R_height = camera_R.getHeight()
    camera_R_lidar = values_lidar[96 : 160]     # values_lidar[470:511] + values_lidar[0:127]

    # ----- Colour -----

    # Update Colour sensor
    colour_data = colour.getImage()

    colour_data = [colour.imageGetRed(colour_data, 1, 0, 0), colour.imageGetGreen(colour_data, 1, 0, 0), colour.imageGetBlue(colour_data, 1, 0, 0)]


def node2position(x,y):
    return x * node_length, y * node_length

def position2node(x,y):
    return np.round( ( x)/node_length), np.round((y)/node_length)

def position2wall(x,y):
    return np.floor((x+0.01)*50)/50, np.floor((y+0.01)*50)/50

def position2tile(x, y, type = "none"):
    if type == "tile":
        return int(2*np.round((x)/tile_length) + initial_tile), int(2*np.round((y)/tile_length) + initial_tile)
    elif type == "transition":
        return int(np.round(2 * ((x)/tile_length - 1/2)) + 1 + initial_tile), int(np.round(2*((y)/tile_length - 1/2)) + 1 + initial_tile)
    elif type == "vertical":
        return int(2*np.round(((x)/tile_length - 1/2))+1 + initial_tile), int(2*np.round((y)/tile_length) + initial_tile)
    elif type == "horizontal":
        return int(2*np.round((x)/tile_length) + initial_tile), int(2*np.round(((y)/tile_length - 1/2))+1 + initial_tile)
    
def tile2node(x,y, type = "none"):
    if type == "tile":
        return tile_length/node_length*np.round((x - initial_tile)/2), tile_length/node_length*np.round((y - initial_tile)/2)
    else:
        return tile_length/node_length*np.round((x - initial_tile - 1)/2), tile_length/node_length*np.round((y - initial_tile - 1)/2)

def tile2position(x,y, type = "none"):
    if type == "tile":
        return (x - initial_tile) * tile_length / 2, (y - initial_tile) * tile_length / 2 
    else:
        return (x - initial_tile - 1) * tile_length /2 + 1/2, (y - initial_tile - 1) * tile_length/2 + 1/2
    
def add_node(v, type = "none"):
    global graph_map, interior, borders, margin 

    if v not in graph_map:
        graph_map[v]={}

        if type == "interior":
            interior.append(v)
        elif type == "border":
            borders.append(v)
        elif type == "margin":
            margin.append(v)

    else:
        if type == "margin":
            if v in margin:
                return
            elif v in interior:
                interior.remove(v)
                margin.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))
            elif v in borders:
                borders.remove(v)
                margin.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))
            else:
                margin.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))

        elif type == "interior":
            if v in margin:
                return
            elif v in interior:
                return
            elif v in borders:
                borders.remove(v)
                interior.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))
            else:
                interior.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))
        
        elif type == "border":
            if v in margin:
                return
            elif v in interior:
                return
            elif v in borders:
                return
            else:
                borders.append(v)
                for u in graph_map[v]:
                    add_edge(v, u, distance_graph(v, u))
            
def add_edge(v1,v2, weight):
    if v1 not in graph_map:
        add_node(v1)
    if v2 not in graph_map:
        add_node(v2)

    graph_map[v1][v2] = weight
    graph_map[v2][v1] = weight

def remove_edge(v1,v2):
    if v1 in graph_map and v2 in graph_map:
        if v2 in graph_map[v1]:
            del graph_map[v1][v2]
            del graph_map[v2][v1]
            
def distance(p, q):
    return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def distance_graph(p, q):
    if p in holes or q in holes:
        return 1e7
    if p in holes or q in swamps:
        return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) * 5
    
    if p in margin and q in margin:
        return 1e5
    elif p in margin or q in margin:
        if p in borders or q in borders:
            return 1e6

        if p[0] == q[0] or p[1] == q[1]:
            return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) * 20
        else:
            return 1e6
    else:
        return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) 

def update_graph():
        
    # Update vertexes'
        add_node(current_node, type = "interior")
        
        interim_nodes = []
        ## Update inner ring
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if i == 0 and j == 0:
                    continue

                target_node = (current_node[0] + i, current_node[1] + j)

                add_node(target_node, type = "interior")

                add_edge(current_node, target_node, distance_graph(current_node, target_node))
                interim_nodes.append(target_node) 

        ## Update outer ring
        for interim_node in interim_nodes:

            for i in range(-1,2,1):
                for j in range(-1,2,1):
                    if i == 0 and j == 0:
                        continue

                    target_node = (interim_node[0] + i, interim_node[1] + j)
                    
                    add_node(target_node, type = "border")
                    add_edge(interim_node, target_node, distance_graph(interim_node, target_node))
                        
def get_distance(token):
    current_distance = float("inf")

    if not token:
        return current_distance

    if token == 'P':
        for potential_token in ['H', 'S', 'U', 'F', 'C', 'O', 'P']:
            for token_position in collected_tokens[potential_token]:
                if current_distance > np.sqrt((token_position[0] - current_position[0]) ** 2 + (token_position[1] - current_position[1]) ** 2):
                    current_distance = np.sqrt((token_position[0] - current_position[0]) ** 2 + (token_position[1] - current_position[1]) ** 2)
    else:
        for token_position in collected_tokens[token]:
            if current_distance > np.sqrt((token_position[0] - current_position[0]) ** 2 + (token_position[1] - current_position[1]) ** 2):
                current_distance = np.sqrt((token_position[0] - current_position[0]) ** 2 + (token_position[1] - current_position[1]) ** 2)
    return current_distance

def dijkstra(current_mood = "none"):
    # Get current node
    graph_distance = {}
    predecessor = {}

    for u in graph_map:
        graph_distance[u] = float('inf')
        predecessor[u] = None
    
    graph_distance[current_node] = 0
    predecessor[current_node] = current_node

    queue = list()
    queue.append(current_node)

    target_node = -1
    
    while queue:
        queue.sort(key=lambda x: graph_distance[x])
        u = queue.pop(0)

        if current_mood == "exploration_mood":
            if u in borders:
                target_node = u
                break
        elif current_mood == "evacuation_mood":
            if u[0] == 0 and u[1] == 0:
                target_node = u
                break

        for v in graph_map[u]:
            if graph_distance[v] > graph_distance[u] + graph_map[u][v]:
                graph_distance[v] = graph_distance[u] + graph_map[u][v]
                predecessor[v] = u
                queue.append(v)            
    
    # Get the target node
    #print("                 Target node = ", target_node)
    #print("                 Target position = ", node2position(target_node[0], target_node[1]))
    if target_node == -1:
        return [], -1, float('inf')
    
    target_distance = 0
    path_to_target = [target_node]
    interim_node = target_node
    while predecessor[interim_node] != current_node:
        target_distance += graph_map[interim_node][predecessor[interim_node]]
        path_to_target.append(predecessor[interim_node])
        interim_node = predecessor[interim_node]
    path_to_target.reverse()

    return path_to_target, target_node, target_distance

def send_map():
    global send_map_flag

    convert_map()
    subMatrix = np.array(full_map)
    ## Get shape
    s = subMatrix.shape
    ## Get shape as bytes
    s_bytes = struct.pack('2i',*s)

    ## Flattening the matrix and join with ','
    flatMap = ','.join(subMatrix.flatten())
    ## Encode
    sub_bytes = flatMap.encode('utf-8')

    ## Add togeather, shape + map
    a_bytes = s_bytes + sub_bytes

    ## Send map data
    emitter.send(a_bytes)

    #STEP3 Send map evaluate request
    map_evaluate_request = struct.pack('c', b'M')
    emitter.send(map_evaluate_request)

    #STEP4 Send an Exit message to get Map Bonus
    ## Exit message
    exit_mes = struct.pack('c', b'E')
    emitter.send(exit_mes)

    print("                 Sending map")
    send_map_flag = True
    return

def send_token(parameter):

    token = parameter[0]

    # Get the current gps position of the robot and convert from meters to cm
    x = int((current_position[0]+initial_position[0])*100)
    y = int(-(current_position[1]+initial_position[1]) * 100)

    #print("                 Sending token: ", token, " at position: ", x, y)

    # Send the message
    tokenType = bytes(token, "utf-8")
    message = struct.pack("i i c", x, y, tokenType)
    emitter.send(message)

    # Add token to the list of collected tokens
    if parameter[1] == "L":
       token_position = [current_position[0] + (np.max(camera_L_lidar)) * np.cos(current_angle + np.pi/2), current_position[1] + (np.max(camera_L_lidar)) * np.sin(current_angle + np.pi/2)]
       collected_tokens[token].append([token_position[0], token_position[1]])
       token_tile = position2tile(token_position[0], token_position[1], type = "transition")
       update_map("token", token_tile, token)
    else:
        token_position = [current_position[0] + (np.max(camera_R_lidar)) * np.cos(current_angle -np.pi/2), current_position[1] + (np.max(camera_R_lidar)) * np.sin(current_angle-np.pi/2)]
        collected_tokens[token].append([token_position[0], token_position[1]])
        token_tile = position2tile(token_position[0], token_position[1], type = "transition")
        update_map("token", token_tile, token)  
    return 0
 
def recognize_victim(image_data, width, height):
    global tick

    img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    img_bin = (img_gray > 200).astype(np.uint8) * 255

    # Save img_bin to a file for each 50 ticks
    # cv2.imwrite(f"img_bin_{tick}.png", img_bin)
    # cv2.imwrite(f"img_gray_{tick}.png", img_gray)
    # cv2.imwrite(f"img_{tick}.png", img)
    #print("                 tick = ", tick, " distancia da direita = ", np.max(camera_R_lidar))


    # Starts from the top left corner and, if it is a black pixel, changes it to white and DFS to its neighbors
    start = (0, 0)
    stack = [start]
    while stack:
        y, x = stack.pop()
        if img_bin[y, x] == 0:
            img_bin[y, x] = 255
            for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= ny < img_bin.shape[0] and 0 <= nx < img_bin.shape[1] and img_bin[ny, nx] == 0:
                    stack.append((ny, nx))

    top, bottom, left, right = -1, -1, -1, -1
    for y in range(0, height):
        if np.any(img_bin[y, :] == 0):
            top = y
            break
    for y in range(height - 1, 0, -1):
        if np.any(img_bin[y, :] == 0):
            bottom = y
            break
    for x in range(0, width):
        if np.any(img_bin[:, x] == 0):
            left = x
            break
    for x in range(width - 1, 0, -1):
        if np.any(img_bin[:, x] == 0):
            right = x
            break

    if top == -1 or bottom == -1 or left == -1 or right == -1:
        return False
    
    middle_x = (left + right) // 2
    middle_y = (top + bottom) // 2

    if (left == 0 or right == width - 1) and (np.abs(left - right) < 20):
        return False
    
    #print("                 top = ", top, " bottom = ", bottom, " middle_x = ", middle_x, " middle_y = ", middle_y)
    #print("                 ", img_gray[top, middle_x], img_gray[middle_y, middle_x], img_gray[bottom, middle_x])
    if img_gray[top, middle_x] <= 200 and img_gray[middle_y, middle_x] <= 200 and img_gray[bottom, middle_x] <= 200:
        return 'S'
    elif img_gray[top, middle_x] > 200 and img_gray[middle_y, middle_x] > 200 and img_gray[bottom, middle_x] <= 200:
        return 'U'
    elif img_gray[top, middle_x] > 200 and img_gray[middle_y, middle_x] <= 200 and img_gray[bottom, middle_x] > 200:
        return 'H'
        
    return False


def recognize_hazard(image_data, width, height):
    global tick
    color_distance = 90

    img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))

    img_dist_red_F = np.linalg.norm(img[..., :3] - [85, 24, 197], axis = -1)
    img_dist_red_O = np.linalg.norm(img[..., :3] - [77, 0, 194], axis = -1)
    img_dist_yellow_O = np.linalg.norm(img[..., :3] - [0, 184, 198], axis = -1)
    img_dist_black_C = np.linalg.norm(img[..., :3] - [25, 25, 25], axis = -1)
    img_dist_white_P = np.linalg.norm(img[..., :3] - [204, 204, 204], axis = -1)
    img_dist_black_WC = np.linalg.norm(img[..., :3] - [32, 30, 18], axis = -1)
    img_dist_black_WC2 = np.linalg.norm(img[..., :3] - [37, 37, 37], axis = -1)

    img_dist_white_floor = np.linalg.norm(img[..., :3] - [192, 192, 192], axis = -1)
    img_dist_gray_shadow = np.linalg.norm(img[..., :3] - [37, 37, 37], axis = -1)
    img_dist_gray_wall = np.linalg.norm(img[..., :3] - [32, 30, 18], axis = -1)

    img_bin = (img_dist_red_F < color_distance) | (img_dist_red_O < color_distance) | (img_dist_yellow_O < color_distance) | (img_dist_black_C < color_distance) | (img_dist_white_P < color_distance)  
    img_bin = img_bin & (img_dist_white_floor != 0) & (img_dist_gray_shadow != 0) & (img_dist_gray_wall != 0)
    #img_bin = (img_bin > 0).astype(np.uint8) * 255    

    #cv2.imwrite(f"HAZARD_img_bin_{tick}.png", img_bin)
    #cv2.imwrite(f"HAZARD_img_{tick}.png", img)   

    # If there is some yellow pixel, return 'O'
    if np.any(img_dist_yellow_O < color_distance):
        return 'O'
    # If there is some red pixel, return 'F'
    elif np.any(img_dist_red_F < 45):
        return 'F'

    left, right, top, bottom = -1, -1, -1, -1
    # Get the lowest row with a 1 at the img_bin
    for y in range(0, height):
        if np.any(img_bin[y, :]):
            top = y
            break
    for y in range(height - 1, 0, -1):
        if np.any(img_bin[y, :]):
            bottom = y
            break
    for x in range(0, width):
        if np.any(img_bin[:, x]):
            left = x
            break
    for x in range(width - 1, 0, -1):
        if np.any(img_bin[:, x]):
            right = x
            break

    if top == -1 or bottom == -1 or left == -1 or right == -1:
        return False
    
    if (left == 0 or right == width - 1) and (np.abs(left - right) < 20):
        return False
    
    # The target the point at the middle, 2/3 bottom, 1/3 top 
    middle_x = (left + right) // 2
    bottom_middle_y = (2 * bottom  + top) // 3

    # If  is black, return 'C'
    if img_dist_black_C[bottom_middle_y, middle_x] < 3:
        return 'C'
    # If the point at the middle, 1/3 bottom, 2/3 top is white, return 'P'
    elif img_dist_white_P[bottom_middle_y, middle_x] < color_distance:
        return 'P'

    return False

def recognize_ambiguity(image_data, width, height, victim, hazard, direction):
    
    img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img_bin = (img_gray > 200).astype(np.uint8) * 255

    # Get top and bottom and left and right 
    top, bottom, left, right = -1, -1, -1, -1
    for y in range(0, height):
        if np.any(img_bin[y, :]):
            top = y
            break
    for y in range(height - 1, 0, -1):
        if np.any(img_bin[y, :]):
            bottom = y
            break
    for x in range(0, width):
        if np.any(img_bin[:, x]):
            left = x
            break
    for x in range(width - 1, 0, -1):
        if np.any(img_bin[:, x]):
            right = x
            break
    
    # Cut the image
    img_bin = img_bin[top:bottom, left:right]

    white_pixels = np.mean(img_bin == 255)*width*height
    if direction == "L":
        density = white_pixels
    else:
        density = white_pixels

    #print("                 Density = ", density, " -> ", hazard, victim)

    if density < 1400:
        #print("                 Ambiguity detected: ", victim, hazard, " -> ", hazard)
        return hazard
    else:
        #print("                 Ambiguity detected: ", victim, hazard, " -> ", victim)
        return victim
    
def get_next_token():
    
    # Victim recognition
    victim_L = False
    victim_R = False
    hazard_L = False
    hazard_R = False    
    if np.max(camera_L_lidar) < 0.1:
        victim_L = recognize_victim(camera_L_data, camera_L_width, camera_L_height)
        hazard_L = recognize_hazard(camera_L_data, camera_L_width, camera_L_height)
    if np.max(camera_R_lidar) < 0.1:
        victim_R = recognize_victim(camera_R_data, camera_R_width, camera_R_height) 
        hazard_R = recognize_hazard(camera_R_data, camera_R_width, camera_R_height)   
          
    # Get the quantity of non-false tokens
    if victim_L != False and hazard_L != False:
        token = recognize_ambiguity(camera_L_data, camera_L_width, camera_L_height, victim_L, hazard_L, "L")
        if (get_distance(token) > 0.3):
            print("                 Ambiguity detected: ", victim_L, hazard_L, " -> ", token)
            return token, "L"
    elif victim_L != False:
        if (get_distance(victim_L) > 0.3):
            print("                 Victim detected: ", victim_L)
            return victim_L, "L"
    elif hazard_L != False:
        if (get_distance(hazard_L) > 0.3):
            print("                 Hazard detected: ", hazard_L)
            return hazard_L, "L"
    
    if victim_R != False and hazard_R != False:
        token = recognize_ambiguity(camera_R_data, camera_R_width, camera_R_height, victim_R, hazard_R, "R")
        if (get_distance(token) > 0.3):
            print("                 Ambiguity detected: ", victim_R, hazard_R, " -> ", token)
            return token, "R"
    elif victim_R != False:
        if(get_distance(victim_R) > 0.3):
            print("                 Victim detected: ", victim_R)
            return victim_R, "R"
    elif hazard_R != False:
        if(get_distance(hazard_R) > 0.3):
            print("                 Hazard detected: ", hazard_R)
            return hazard_R, "R"
    
    return False       

def get_next_action():
    global action_parameter, hole_flag
    global continue_send_token_flag
    global target_position
    global target_node, old_node
    global current_mood

    if tick <= 10:
        return "walk", "none"
    elif tick <= 20:
        return "stop", "none"
    
    if tick % 500 == 0:
        if old_node == current_node:
            return "hard_reset", "none"
        old_node = current_node
        
    if send_map_flag == True:
        return "stop", "none"

    if current_mood == "evacuation_mood" and distance(current_node, [0,0]) < 0.03:
        print("                 Evacuation completed")
        return "send_map", "none"

    if hole_flag == True:
       # print("                 Hole detected")   
        hole_flag = False 
        return "avoid_holes", "none"
    
    if action_type == "point_token":
        return "depart_token", "none"
    
    if action_type == "depart_token":
        return "scan_token", 0
    
    if action_type == "scan_token":
        if action_parameter == 100:
            return "stop", "none"
        potential_token = get_next_token()
        if potential_token != False:
            return "stop_and_send_token", potential_token  
        else:
            return "scan_token", action_parameter + 1

    if action_type == "stop_and_send_token":
        potential_token = get_next_token()
        if potential_token != False:
            return "send_token", potential_token   
    
    if tick_buffer == 0:
        potential_token  = get_next_token()
        if potential_token != False:
            return "stop_and_send_token", potential_token
    
    # Navigation
    path_to_target, target_node, target_distance = dijkstra(current_mood)

    if target_distance >= 1e6:
        print("                 No path found, going to the initial node")
        current_mood = "evacuation_mood"
        path_to_target, target_node, target_distance = dijkstra(current_mood)

    target_position = node2position(path_to_target[0][0], path_to_target[0][1]) 
    return "go_to", [target_position[0], target_position[1]]

    return "stop", "none"

# Action = [ Type, Parameter]
def execute_action(action_type, parameter):
    global tick_action
    global tick_buffer
    global tick_continue
    global continue_send_token_flag
    global target_position
    
    if action_type == "avoid_holes":
        wheel_L.setVelocity(-max_velocity)
        wheel_R.setVelocity(-max_velocity)
        tick_action = 20

    if action_type == "point_token": 
        if parameter[1] == "L":
            wheel_L.setVelocity(max_velocity)
            wheel_R.setVelocity(-max_velocity)
            tick_action = 5
        else:
            wheel_L.setVelocity(-max_velocity)
            wheel_R.setVelocity(max_velocity)
            tick_action = 5

    if action_type == "depart_token":
        wheel_L.setVelocity(-max_velocity)
        wheel_R.setVelocity(-max_velocity)
        tick_action = 0

    if action_type == "scan_token":
        wheel_L.setVelocity(max_velocity/2)
        wheel_R.setVelocity(-max_velocity/2)
        parameter = parameter + 1
        tick_action = 1

    if action_type == "stop_and_send_token":   
        wheel_L.setVelocity(0)
        wheel_R.setVelocity(0)
        tick_action = 100

    if action_type == "send_token":
        send_token(parameter)
        tick_action = 5

    if action_type == "send_map" or tick == 33125:
        send_map()
        tick_action = 5

    if action_type == "go_to":
        target_angle = np.arctan2(parameter[1] - current_position[1], parameter[0] - current_position[0]) 
        target_distance = np.sqrt((parameter[0]  - current_position[0]) ** 2 + (parameter[1]  - current_position[1]) ** 2) 

        if target_distance < 0.03:
           # print("                 Target reached")
            wheel_L.setVelocity(0) 
            wheel_R.setVelocity(0)

        delta_angle = target_angle - current_angle
        if delta_angle > np.pi:
            delta_angle = delta_angle - 2 * np.pi
        elif delta_angle < -np.pi:
            delta_angle = delta_angle + 2 * np.pi
        
       # print("                 target_angle = ", target_angle, " current_angle = ", current_angle, " delta_angle = ", delta_angle)
       # print("                 target_distance = ", target_distance)

        # If delta_angle /2greater than 30 degrees, turn left
        if delta_angle > 15 * np.pi / 180:
            wheel_L.setVelocity(-max_velocity)
            wheel_R.setVelocity(max_velocity)
            tick_action = 1
        elif delta_angle < - 15* np.pi / 180:
            wheel_L.setVelocity(max_velocity)
            wheel_R.setVelocity(-max_velocity)
            tick_action = 1
        else:
            wheel_L.setVelocity(max_velocity)
            wheel_R.setVelocity(max_velocity)
            tick_action = 10
    
    elif action_type == "stop":
        wheel_L.setVelocity(0)
        wheel_R.setVelocity(0)
        tick_action = 80

    elif action_type == "walk":
        wheel_L.setVelocity(max_velocity)
        wheel_R.setVelocity(max_velocity)
        tick_action = 15
    
    elif action_type == "hard_reset":
        wheel_L.setVelocity(-max_velocity)
        wheel_R.setVelocity(-max_velocity)
        tick_action = 10
        
            
def update_map(tile_type, tile_position, tile_parameter):
    global compact_map

    #print("                 Update map: ", tile_type, tile_position, tile_parameter)

    if tile_type == "area4":
        compact_map[tile_position[0]][tile_position[1]] = '*'
    elif tile_type == "walls":
        if(compact_map[tile_position[0]][tile_position[1]] == '0'):
            compact_map[tile_position[0]][tile_position[1]] = '1'  
    elif tile_type == "holes":
        compact_map[tile_position[0]][tile_position[1]] = '2' 
    elif tile_type == "swamp":
        compact_map[tile_position[0]][tile_position[1]] = '3'
    elif tile_type == "checkpoints":
        compact_map[tile_position[0]][tile_position[1]] = '4'
    elif tile_type == "initial_position":
        compact_map[initial_tile][initial_tile] = '5'
    elif tile_type == "transition_areas":
        compact_map[tile_position[0]][tile_position[1]] = tile_parameter
    elif tile_type == "token":
        compact_map[tile_position[0]][tile_position[1]] = tile_parameter

def convert_map():
    global compact_map, full_map, aux_map

    top = -1
    bottom = -1
    left = -1
    right = -1

    # Remove empty the first and last empty rows
    for y in range(0, max_tile):
        if np.any(compact_map[:, y] != '0'):
            bottom = y
            break
    for y in range(max_tile - 1, 0, -1):
        if np.any(compact_map[:, y] != '0'):
            top = y
            break
    for x in range(0, max_tile):
        if np.any(compact_map[x, :] != '0'):
            left = x
            break
    for x in range(max_tile - 1, 0, -1):
        if np.any(compact_map[x, :] != '0'):
            right = x
            break

    for y in range(top, bottom-2, -1):
        for x in range(left, right+1):
            print(compact_map[x][y], end = ' ')
            aux_map[max_tile-1-y][x] = compact_map[x][y]
        print("\n")

    full_map = np.full((2*max_tile, 2*max_tile), '0', dtype = str)

    for x in range(0, max_tile):
        for y in range(0, max_tile):
            
            # Full tile
            if x % 2 == 1 and y % 2 == 1: 
                for i in range(-1,2,1):
                    for j in range(-1,2,1):
                        if np.abs(i) + np.abs(j) == 2:
                            full_map[2*x + i][2*y + j] = aux_map[x][y]
                        
                        if aux_map[x][y] == '*':
                            full_map[2*x + i][2*y + j] = '*'
            # # Vertical transition area 4
            # elif x % 2 == 1 and y % 2 == 0 and current_area == 4:
            #     full_map[2*x][2*y] = aux_map[x][y]
            #     if aux_map[x][y] != '0':
            #         full_map[2*x - 1][2*y] = '*'
            #         full_map[2*x + 1][2*y] = '*'
            # # Horizontal transition area 4
            # elif x % 2 == 0 and y % 2 == 1 and current_area == 4:
            #     full_map[2*x][2*y] = aux_map[x][y]
            #     if aux_map[x][y] != '0':
            #         full_map[2*x][2*y - 1] = '*'
            #         full_map[2*x][2*y + 1] = '*'
            
            # Horizontal transition
            elif x % 2 == 0 and y % 2 == 1:
                full_map[2*x][2*y] = aux_map[x][y]
                if aux_map[x][y] != '0':
                    full_map[2*x][2*y - 1] = '1'
                    full_map[2*x][2*y + 1] = '1'
            
            # Vertical transition
            elif x % 2 == 1 and y % 2 == 0:
                full_map[2*x][2*y] = aux_map[x][y]
                if aux_map[x][y] != '0':
                    full_map[2*x - 1][2*y] = '1'
                    full_map[2*x + 1][2*y] = '1'
            

            
            else:
                full_map[2*x][2*y] = aux_map[x][y]

    top = -1
    bottom = -1
    left = -1
    right = -1
    for y in range(0, 2*max_tile-1):
        if np.any(full_map[y, :] != '0'):
            top = y
            break
    for y in range(2*max_tile-1 - 1, 0, -1):
        if np.any(full_map[y, :] != '0'):
            bottom = y
            break
    for x in range(0, 2*max_tile-1):
        if np.any(full_map[:, x] != '0'):
            left = x
            break
    for x in range(2*max_tile-1 - 1, 0, -1):
        if np.any(full_map[:, x] != '0'):
            right = x
            break

    for x in range(top, bottom+1):
        for y in range(left, right+1):
            print(full_map[x][y], end = ' ')
        print("\n")     

    full_map = full_map[top:bottom+1, left:right+1]    

# def debug_print():
#     print("######################")
#     print("Tick: ", tick)
#     print("     Posicao = ", current_position, "Posicao inicial = ", initial_position)
#     print("     Angulo = ", current_angle)
#     print("     Action = ", action_type, action_parameter)
#     print("         current_node = ", current_node)
#     print("         collected_tokens = ", collected_tokens)
#     print("         current_area = ", current_area)
#     print("     Colour data = ", colour_data)
#     #print("     Holes = ", holes)
#     #print("     Swamps = ", swamps)
#     #print("     Areas = ", transition_areas)
#     #print("     Checkpoints = ", checkpoints)
#     print("     camera_L_lidar = ", np.max(camera_L_lidar))
#     print("     camera_R_lidar = ", np.max(camera_R_lidar))
    
# ==================== MAIN LOOP ====================
start = robot.getTime()

while robot.step(timeStep) != -1: 
    
    # ----- Initialization -----

    update_sensors()

    if(tick >= 20):
        if(tick % 20 == 0):
           # print("             update_sensors")
            update_walls()
            update_graph()
        update_holes()
        update_areas()
        update_swamps()
        update_checkpoints()
        

    
        
    
        # img_L = np.frombuffer(camera_L_data, dtype=np.uint8).reshape((64, 64, 4))
        # img_L_gray = np.dot(img_L[..., :3], [0.2989, 0.5870, 0.1140])
        # img_L_bin = (img_L_gray > 200).astype(np.uint8) * 255
        # cv2.imwrite(f"img_gray_{tick}_L.png", img_L_gray)
        # cv2.imwrite(f"img_bin_{tick}_L.png", img_L_bin)
        # cv2.imwrite(f"img_{tick}_L.png", img_L)
        # img_R = np.frombuffer(camera_R_data, dtype=np.uint8).reshape((64, 64, 4))
        # img_R_gray = np.dot(img_R[..., :3], [0.2989, 0.5870, 0.1140])
        # img_R_bin = (img_R_gray > 200).astype(np.uint8) * 255
        # cv2.imwrite(f"img_gray_{tick}_R.png", img_R_gray)
        # cv2.imwrite(f"img_bin_{tick}_R.png", img_R_bin)
        # cv2.imwrite(f"img_{tick}_R.png", img_R)

    # if tick % 50 == 0 and tick > 0:
    #     save_plot()

    if(tick < 20):
        first_ticks()

    # ----- Main -----
    if tick_buffer > 0:
        tick_buffer = tick_buffer - 1

    if tick_continue > 0:
        tick_continue = tick_continue - 1

    if tick_areas > 0:
        tick_areas = tick_areas - 1
    if tick_swamp > 0:
        tick_swamp = tick_swamp - 1
    if tick_checkpoint > 0:
        tick_checkpoint = tick_checkpoint - 1
    if tick_holes > 0:
        tick_holes = tick_holes - 1

    if tick_action > 0:
        tick_action = tick_action - 1
    else:
        action_type, action_parameter = get_next_action()
        execute_action(action_type, action_parameter)
    # print("colour_data = ", colour_data)
    tick = tick + 1 