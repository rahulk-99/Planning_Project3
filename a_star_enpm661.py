# -*- coding: utf-8 -*-
# ===========================
# Movement Functions for A*
# ===========================
def move_right(node):
    """Moves the node one step to the right (positive x-direction)."""
    return (node[0] + 1, node[1])

def move_left(node):
    """Moves the node one step to the left (negative x-direction)."""
    return (node[0] - 1, node[1])

def move_up(node):
    """Moves the node one step up (positive y-direction)."""
    return (node[0], node[1] + 1)

def move_down(node):
    """Moves the node one step down (negative y-direction)."""
    return (node[0], node[1] - 1)

def move_up_right(node):
    """Moves the node diagonally up-right (positive x and y)."""
    return (node[0] + 1, node[1] + 1)

def move_up_left(node):
    """Moves the node diagonally up-left (negative x, positive y)."""
    return (node[0] - 1, node[1] + 1)

def move_down_right(node):
    """Moves the node diagonally down-right (positive x, negative y)."""
    return (node[0] + 1, node[1] - 1)

def move_down_left(node):
    """Moves the node diagonally down-left (negative x and y)."""
    return (node[0] - 1, node[1] - 1)

# Dictionary to Map Movement Actions to Corresponding Functions
actions = {
    "right": move_right,        # Move right (+x)
    "left": move_left,          # Move left (-x)
    "up": move_up,              # Move up (+y)
    "down": move_down,          # Move down (-y)
    "up_right": move_up_right,  # Move diagonally up-right (+x, +y)
    "up_left": move_up_left,    # Move diagonally up-left (-x, +y)
    "down_right": move_down_right,  # Move diagonally down-right (+x, -y)
    "down_left": move_down_left     # Move diagonally down-left (-x, -y)
}

# Function to Generate Valid Movement Nodes
def get_valid_moves(node):
    """
    Returns a list of new valid positions after applying all movement actions.
    Parameters:
    node (tuple): The current position (x, y).
    Returns:
    list: A list of tuples representing all possible next positions.
    """
    return [action(node) for action in actions.values()]

import cv2
import numpy as np
import queue
import os
import shutil
import matplotlib.pyplot as plt
import math

# Define canvas dimensions
canvas_width = 600
canvas_height = 250

# Function to check if a point is inside the number "1"
def is_inside_1(x, y, x0=390+60, y0=145, width=10, height=40, thickness=3):
    """
    Checks if a point (x, y) is inside the number "1" using half-planes.
    """
    # Vertical bar of "1"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    return False


clearance = 5

# Function to check if a point is inside the second "6"
def is_inside_6_second(x, y, x0=340+60, y0=145, large_radius=20, medium_radius=14,
                      small_radius=8, hole_radius=5, thickness=3):
    """
    Defines the second "6" using a series of arcs with a small hole in the center.
    """
    top_x = x0 + medium_radius // 2
    top_y = y0 - large_radius * 1.5
    inside_top = ((x - top_x) * 2 + (y - top_y) * 2) <= small_radius ** 2 and x >= top_x

    mid_x = x0 + large_radius - thickness
    mid_y = y0 - large_radius
    inside_middle = ((x - mid_x) * 2 + (y - mid_y) * 2) <= large_radius ** 2 and x <= mid_x

    bottom_x = x0 + medium_radius
    bottom_y = y0 - medium_radius
    inside_bottom = ((x - bottom_x) * 2 + (y - bottom_y) * 2) <= medium_radius ** 2 and x >= bottom_x + thickness

    hole_x = x0 + medium_radius
    hole_y = y0 - medium_radius
    inside_hole = ((x - hole_x) * 2 + (y - hole_y) * 2) <= hole_radius ** 2

    if (inside_top or inside_middle or inside_bottom) and not inside_hole:
        return True
    return False

# Function to check if a point is inside the first "6"
def is_inside_6_first(x, y, x0=285+60, y0=145, large_radius=20, medium_radius=14,
                     small_radius=8, hole_radius=5, thickness=3):
    """
    Defines the first "6" using a similar arc-based structure as the second "6".
    """
    top_x = x0 + medium_radius // 2
    top_y = y0 - large_radius * 1.5
    inside_top = ((x - top_x) * 2 + (y - top_y) * 2) <= small_radius ** 2 and x >= top_x

    mid_x = x0 + large_radius - thickness
    mid_y = y0 - large_radius
    inside_middle = ((x - mid_x) * 2 + (y - mid_y) * 2) <= large_radius ** 2 and x <= mid_x

    bottom_x = x0 + medium_radius
    bottom_y = y0 - medium_radius
    inside_bottom = ((x - bottom_x) * 2 + (y - bottom_y) * 2) <= medium_radius ** 2 and x >= bottom_x + thickness

    hole_x = x0 + medium_radius
    hole_y = y0 - medium_radius
    inside_hole = ((x - hole_x) * 2 + (y - hole_y) * 2) <= hole_radius ** 2

    if (inside_top or inside_middle or inside_bottom) and not inside_hole:
        return True
    return False

# Function to check if a point is inside the letter "M"
def is_inside_M(x, y, x0=230+60, y0=145, width=30, height=40, thickness=3):
    """
    Defines the letter "M" using vertical bars and diagonal slopes.
    """
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True

    slope_left = (height / 2) / (width / 2 - thickness)
    y_expected_left = slope_left * (x - x0 - thickness) + (y0 - height)
    if x0 + thickness <= x <= x0 + width / 2 and y_expected_left <= y <= y_expected_left + 10:
        return True

    slope_right = (-height / 2) / (width / 2 - thickness)
    y_expected_right = slope_right * (x - (x0 + width / 2)) + (y0 - height / 2)
    if x0 + width / 2 <= x <= x0 + width - thickness and y_expected_right <= y <= y_expected_right + 10:
        return True

    return False

# Function to check if a point is inside the letter "P"
def is_inside_P(x, y, x0=200+60, y0=145, width=20, height=40, thickness=3):
    """
    Defines the letter "P" using a vertical bar and a semi-circular top.
    """
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True

    curve_x_center = x0 + thickness
    curve_y_center = y0 - (3 * height / 4)
    curve_radius = height / 4
    if ((x - curve_x_center) * 2 + (y - curve_y_center) * 2) <= curve_radius ** 2 and y < y0 - height // 2 and x > x0 + thickness:
        return True

    return False

# Function to check if a point is inside the letter "N"
def is_inside_N(x, y, x0=160+60, y0=145, width=20, height=40, thickness=3):
    """
    Defines the letter "N" using two vertical bars and a diagonal.
    """
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True

    slope = (2/3)*height / (width - 2 * thickness)
    y_expected = (slope * (x - (x0 + thickness))) + (y0 - height)
    if x0 + thickness <= x <= x0 + width - thickness and y_expected <= y <= y_expected + height/3:
        return True

    return False

# Function to check if a point is inside the letter "E"
def is_inside_E(x, y, x0=120+60, y0=145, width=20, height=40, mid_width=15, thickness=3):
    """
    Defines the letter "E" using three horizontal bars and a vertical bar.
    """
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 <= x <= x0 + width and y0 - height <= y <= y0 - height + thickness:
        return True
    if x0 <= x <= x0 + mid_width and y0 - height // 2 - thickness // 2 <= y <= y0 - height // 2 + thickness // 2:
        return True
    if x0 <= x <= x0 + width and y0 - thickness <= y <= y0:
        return True

    return False

# Define clearance around the obstacle
clearance = 5

def generate_obstacles(grid_width, grid_height, clearance):
    """
    Generates binary masks for obstacles and clearance regions in a given grid.
    Parameters:
    grid_width (int): Width of the grid (canvas).
    grid_height (int): Height of the grid (canvas).
    clearance (int): Buffer around obstacles representing the clearance.
    Returns:
    tuple: (obstacle_mask, clearance_mask)
    - obstacle_mask: A binary mask where obstacles are marked with 255.
    - clearance_mask: A binary mask where clearance regions are marked with 255, and obstacles remain distinct with a value of 200.
    """
    # Initialize blank grids for obstacles and clearance areas
    obstacle_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    clearance_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Iterate over every pixel in the grid
    for y in range(grid_height):
        for x in range(grid_width):
            # Check if the current pixel lies within any obstacle shape
            if (is_inside_E(x, y) or is_inside_N(x, y) or is_inside_P(x, y) or
                is_inside_M(x, y) or is_inside_6_first(x, y) or
                is_inside_6_second(x, y) or is_inside_1(x, y)):
                obstacle_mask[y, x] = 255  # Mark the obstacle pixel

    # Define a kernel size based on clearance and apply dilation for buffer region
    kernel = np.ones((clearance * 2 + 1, clearance * 2 + 1), np.uint8)
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    # Ensure obstacle pixels remain distinguishable inside the clearance mask
    clearance_mask[np.where(obstacle_mask == 255)] = 200  # Assign distinct value to obstacles

    clearance_mask[0:clearance, :] = 255

    # Bottom boundary
    clearance_mask[grid_height-clearance:grid_height, :] = 255

    # Left boundary
    clearance_mask[:, 0:clearance] = 255

    # Right boundary
    clearance_mask[:, grid_width-clearance:grid_width] = 255

    return obstacle_mask, clearance_mask

# Generate the updated masks
obstacle_mask, clearance_mask = generate_obstacles(canvas_width, canvas_height, clearance)

# A* Algorithm Implementation
def heuristic(node, goal):
    """
    Calculates the Euclidean distance between the current node and the goal.
    """
    return math.sqrt((node[0] - goal[0])*2 + (node[1] - goal[1])*2)

def astar(start, goal, grid_width, grid_height, clearance_mask):
    """
    Performs A* search to find the optimal path from start to goal.

    Parameters:
    start (tuple): (x, y) coordinates of the start position.
    goal (tuple): (x, y) coordinates of the goal position.
    grid_width (int): Width of the grid (canvas).
    grid_height (int): Height of the grid (canvas).
    clearance_mask (numpy array): A binary mask indicating obstacles and clearance regions.

    Returns:
    tuple: (optimal_path, explored_nodes)
    """
    # Priority queue for A* (f_score, node)
    open_set = queue.PriorityQueue()

    # Set to track visited nodes
    closed_set = set()

    # Dictionary to store parent-child relationships for backtracking
    parent_map = {}

    # Track g_score (cost from start to current node)
    g_score = {start: 0}

    # Track f_score (g_score + heuristic)
    f_score = {start: heuristic(start, goal)}

    # List to keep track of explored nodes for visualization
    explored_nodes = []

    # Add start node to the open set
    open_set.put((f_score[start], start))

    # Define possible movement actions (8-connected grid)
    actions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]

    # A* loop
    while not open_set.empty():
        # Get the node with lowest f_score
        _, current = open_set.get()

        # Skip if already processed
        if current in closed_set:
            continue

        # Skip nodes inside obstacles or clearance areas
        if clearance_mask[current[1], current[0]] == 255:
            continue

        # Add to closed set
        closed_set.add(current)

        # Store for visualization
        explored_nodes.append(current)

        # Check if goal reached
        if current == goal:
            # Backtrack to retrieve the optimal path
            optimal_path = []
            while current in parent_map:
                optimal_path.append(current)
                current = parent_map[current]
            optimal_path.append(start)
            optimal_path.reverse()
            return optimal_path, explored_nodes

        # Explore all neighbors
        for dx, dy in actions:
            new_x, new_y = current[0] + dx, current[1] + dy
            neighbor = (new_x, new_y)

            # Check if neighbor is valid
            if (0 <= new_x < grid_width and
                0 <= new_y < grid_height and
                clearance_mask[new_y, new_x] == 0 and
                neighbor not in closed_set):

                # Calculate movement cost (diagonal moves cost more)
                move_cost = 1.4 if (dx != 0 and dy != 0) else 1.0

                # Calculate tentative g_score
                tentative_g = g_score.get(current, float('inf')) + move_cost

                # If this path is better than any previous one
                if tentative_g < g_score.get(neighbor, float('inf')):
                    # Record this path
                    parent_map[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

    # If A* completes without finding a path
    return None, explored_nodes


def visualize_astar(grid_width, grid_height, explored_nodes, optimal_path, clearance_mask, obstacle_mask):
    """
    Uses OpenCV to visualize A* exploration, ensuring obstacle and clearance have distinct colors.
    Parameters:
    grid_width (int): Width of the grid (canvas).
    grid_height (int): Height of the grid (canvas).
    explored_nodes (list): List of nodes that A* explored.
    optimal_path (list): List of nodes forming the shortest path (if found).
    clearance_mask (numpy array): Mask indicating clearance areas (green).
    obstacle_mask (numpy array): Mask indicating obstacles (red).
    Returns:
    str: Path to the generated A* visualization video.
    """
    # Create a blank white canvas (RGB image)
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Draw clearance region in Green
    canvas[np.where(clearance_mask == 255)] = [0, 255, 0]  # Green for clearance

    # Draw obstacle region in Red
    canvas[np.where(obstacle_mask == 255)] = [255, 0, 0]  # Red for obstacles

    # Draw start and goal points
    cv2.circle(canvas, start_position, 1, (0, 0, 0), -1)  # Black for start position
    cv2.circle(canvas, goal_position, 1, (0, 0, 0), -1)  # Black for goal position

    # Create a directory to save frames (for video generation)
    frame_dir = "astar_frames"
    os.makedirs(frame_dir, exist_ok=True)

    frame_index = 0  # Frame counter

    # Plot explored nodes (Blue) and save frames for animation
    for i, node in enumerate(explored_nodes):
        cv2.circle(canvas, node, 1, (0, 0, 255), -1)  # Blue for explored nodes
        cv2.imwrite(f"{frame_dir}/frame_{frame_index:04d}.png", canvas)
        frame_index += 1

    # Handle case where no valid path is found
    if optimal_path == None:
        print("No valid path found. Please rerun the code with valid goal coordinates. (0,0) coordinates are out of index!")
        return None

    # Plot the optimal shortest path (Yellow) and save frames
    for i, node in enumerate(optimal_path):
        cv2.circle(canvas, node, 2, (0, 255, 255), -1)  # Yellow for the optimal path
        cv2.imwrite(f"{frame_dir}/frame_{frame_index:04d}.png", canvas)
        frame_index += 1

    # Convert saved frames into a video file
    video_path = "astar.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for video writing
    video = cv2.VideoWriter(video_path, fourcc, 400, (grid_width, grid_height))  # 400 FPS video

    # Read and write frames into the video
    for i in range(frame_index):
        frame = cv2.imread(os.path.join(frame_dir, f"frame_{i:04d}.png"))
        video.write(frame)

    # Release the video writer
    video.release()
    return video_path

# Prompt user for valid start position
while True:
    start_x = int(input("Enter X coordinate of start position: "))
    start_y = int(input("Enter Y coordinate of start position: "))
    start_orientation = int(input("Enter orientation of start position: "))

    # Check if the start position is out of canvas bounds
    if start_x < 0 or start_y < 0 or start_x >= canvas_width or start_y >= canvas_height:
        print("Start position is outside the canvas. Please enter a new start position.")
        continue

    # Check if the start position is inside an obstacle or its clearance region
    elif clearance_mask[start_y, start_x] == 200 or clearance_mask[start_y, start_x] == 255:
        print("Start position is inside an obstacle. Please enter a new start position.")
        continue
    else:
        break  # Valid input, exit loop

# Prompt user for valid goal position
while True:
    goal_x = int(input("Enter X coordinate of goal position: "))
    goal_y = int(input("Enter Y coordinate of goal position: "))
    goal_orientation = int(input("Enter orientation of goal position: "))

    # Check if the goal position is out of canvas bounds

    if goal_x < 0 or goal_y < 0 or goal_x >= canvas_width or goal_y >= canvas_height:
        print("Goal position is outside the canvas. Please enter a new goal position.")
        continue

    # Check if the goal position is inside an obstacle or its clearance region
    elif clearance_mask[goal_y, goal_x] == 200 or clearance_mask[goal_y, goal_x] == 255:
        print("Goal position is inside an obstacle. Please enter a new goal position.")
        continue
    else:
        break  # Valid input, exit loop

# Store start and goal positions
start_position = (start_x, start_y)
goal_position = (goal_x, goal_y)

# Convert positions to match the bottom-left origin system
start_position = (start_position[0], canvas_height - start_position[1])
goal_position = (goal_position[0], canvas_height - goal_position[1])

# Run A* with mathematically defined obstacles
path_result, explored_nodes = astar(start_position, goal_position, canvas_width, canvas_height, clearance_mask)

# Generate A* visualization using updated obstacle and clearance representation
video_file = visualize_astar(canvas_width, canvas_height, explored_nodes, path_result, clearance_mask, obstacle_mask)
print("A* visualization saved as astar.mp4!")

import matplotlib.pyplot as plt

def plot_final_trajectory(grid_width, grid_height, optimal_path, clearance_mask, obstacle_mask):
    """
    Plots the final A* trajectory using Matplotlib, displaying obstacles, clearance, and the optimal path.
    Parameters:
    - grid_width (int): Width of the grid.
    - grid_height (int): Height of the grid.
    - optimal_path (list of tuples): List of (x, y) coordinates representing the shortest path.
    - clearance_mask (numpy array): Binary mask indicating clearance areas.
    - obstacle_mask (numpy array): Binary mask indicating obstacle areas.
    """
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(9, 5))

    # Flip the clearance and obstacle masks to match bottom-left origin system
    flipped_clearance = cv2.flip(clearance_mask, 0)
    flipped_obstacle = cv2.flip(obstacle_mask, 0)

    # Create a blank white image for visualization
    img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Color the clearance area (green)
    img[np.where(flipped_clearance == 255)] = [0, 255, 0]  # Green for clearance

    # Color the obstacles (red)
    img[np.where(flipped_obstacle == 255)] = [255, 0, 0]  # Red for obstacle

    # Convert the image to RGB format for Matplotlib display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the obstacle and clearance map
    ax.imshow(img)

    # Plot the final trajectory (yellow path)
    if optimal_path:
        x_coords, y_coords = zip(*optimal_path)  # Extract x and y coordinates from the path
        ax.plot(x_coords, [grid_height - y for y in y_coords], color='yellow', linewidth=2, label="Optimal Path")

    # Mark the start and goal positions on the plot
    ax.scatter(start_position[0], grid_height - start_position[1], color='green', s=50, label="Start")  # Green marker for start
    ax.scatter(goal_position[0], grid_height - goal_position[1], color='red', s=50, label="Goal")  # Red marker for goal

    # Set plot title and axis labels
    ax.set_title("Final A* Trajectory")
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")

    # Add a legend to distinguish start, goal, and path
    ax.legend()

    # Invert the y-axis to match coordinate system expectations
    ax.invert_yaxis()

    # Enable grid with dashed lines for better readability
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Display the final trajectory plot
    plt.show()

# Call the function to visualize the A* path after execution
if path_result:
    plot_final_trajectory(canvas_width, canvas_height, path_result, clearance_mask, obstacle_mask)
else:
    print("No valid path found! Please try again with a different goal position")