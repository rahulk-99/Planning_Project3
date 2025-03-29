# -- coding: utf-8 --
"""A* Implementation with Vector Plotting and Video Saving"""

import cv2
import numpy as np
import heapq
import math
import time
import os
import matplotlib.pyplot as plt
import shutil

# ===========================
# Map Generation
# ===========================
canvas_width = 600
canvas_height = 250
clearance = 5

def is_inside_1(x, y, x0=390+60, y0=145, width=10, height=40, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    return False

def is_inside_6_second(x, y, x0=340+60, y0=145, large_radius=20, medium_radius=14,
                      small_radius=8, hole_radius=5, thickness=3):
    top_x = x0 + medium_radius // 2
    top_y = y0 - large_radius * 1.5
    inside_top = ((x - top_x) ** 2 + (y - top_y) ** 2) <= small_radius ** 2 and x >= top_x
    mid_x = x0 + large_radius - thickness
    mid_y = y0 - large_radius
    inside_middle = ((x - mid_x) ** 2 + (y - mid_y) ** 2) <= large_radius ** 2 and x <= mid_x
    bottom_x = x0 + medium_radius
    bottom_y = y0 - medium_radius
    inside_bottom = ((x - bottom_x) ** 2 + (y - bottom_y) ** 2) <= medium_radius ** 2 and x >= bottom_x + thickness
    hole_x = x0 + medium_radius
    hole_y = y0 - medium_radius
    inside_hole = ((x - hole_x) ** 2 + (y - hole_y) ** 2) <= hole_radius ** 2
    if (inside_top or inside_middle or inside_bottom) and not inside_hole:
        return True
    return False

def is_inside_6_first(x, y, x0=285+60, y0=145, large_radius=20, medium_radius=14,
                     small_radius=8, hole_radius=5, thickness=3):
    top_x = x0 + medium_radius // 2
    top_y = y0 - large_radius * 1.5
    inside_top = ((x - top_x) ** 2 + (y - top_y) ** 2) <= small_radius ** 2 and x >= top_x
    mid_x = x0 + large_radius - thickness
    mid_y = y0 - large_radius
    inside_middle = ((x - mid_x) ** 2 + (y - mid_y) ** 2) <= large_radius ** 2 and x <= mid_x
    bottom_x = x0 + medium_radius
    bottom_y = y0 - medium_radius
    inside_bottom = ((x - bottom_x) ** 2 + (y - bottom_y) ** 2) <= medium_radius ** 2 and x >= bottom_x + thickness
    hole_x = x0 + medium_radius
    hole_y = y0 - medium_radius
    inside_hole = ((x - hole_x) ** 2 + (y - hole_y) ** 2) <= hole_radius ** 2
    if (inside_top or inside_middle or inside_bottom) and not inside_hole:
        return True
    return False

def is_inside_M(x, y, x0=230+60, y0=145, width=30, height=40, thickness=3):
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

def is_inside_P(x, y, x0=200+60, y0=145, width=20, height=40, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    curve_x_center = x0 + thickness
    curve_y_center = y0 - (3 * height / 4)
    curve_radius = height / 4
    if ((x - curve_x_center) ** 2 + (y - curve_y_center) ** 2) <= curve_radius ** 2 and y < y0 - height // 2 and x > x0 + thickness:
        return True
    return False

def is_inside_N(x, y, x0=160+60, y0=145, width=20, height=40, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True
    slope = (2/3)*height / (width - 2 * thickness)
    y_expected = (slope * (x - (x0 + thickness))) + (y0 - height)
    if x0 + thickness <= x <= x0 + width - thickness and y_expected <= y <= y_expected + height/3:
        return True
    return False

def is_inside_E(x, y, x0=120+60, y0=145, width=20, height=40, mid_width=15, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 <= x <= x0 + width and y0 - height <= y <= y0 - height + thickness:
        return True
    if x0 <= x <= x0 + mid_width and y0 - height // 2 - thickness // 2 <= y <= y0 - height // 2 + thickness // 2:
        return True
    if x0 <= x <= x0 + width and y0 - thickness <= y <= y0:
        return True
    return False

def generate_obstacles(grid_width, grid_height, clearance):
    obstacle_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    clearance_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for y in range(grid_height):
        for x in range(grid_width):
            if (is_inside_E(x, y) or is_inside_N(x, y) or is_inside_P(x, y) or
                is_inside_M(x, y) or is_inside_6_first(x, y) or
                is_inside_6_second(x, y) or is_inside_1(x, y)):
                obstacle_mask[y, x] = 255
    kernel = np.ones((clearance * 2 + 1, clearance * 2 + 1), np.uint8)
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    clearance_mask[np.where(obstacle_mask == 255)] = 200
    clearance_mask[0:clearance, :] = 255
    clearance_mask[grid_height-clearance:grid_height, :] = 255
    clearance_mask[:, 0:clearance] = 255
    clearance_mask[:, grid_width-clearance:grid_width] = 255
    return obstacle_mask, clearance_mask

obstacle_mask, clearance_mask = generate_obstacles(canvas_width, canvas_height, clearance)

# ===========================
# A* Algorithm Implementation
# ===========================
class Node:
    def _init_(self, x, y, theta, cost_to_come, parent):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost_to_come = cost_to_come
        self.cost_to_go = 0
        self.total_cost = 0
        self.parent = parent

    def _lt_(self, other):
        return self.total_cost < other.total_cost

def get_motion_set(step):
    actions = [0, 30, 60, -30, -60]
    motions = []
    for angle in actions:
        motions.append((step, angle))
    return motions

def is_valid(x, y):
    return 0 <= x < canvas_width and 0 <= y < canvas_height and clearance_mask[int(y), int(x)] == 0

visited = np.zeros((canvas_height * 2, canvas_width * 2, 12), dtype=np.uint8)

def is_visited(x, y, theta):
    i = int(y * 2)
    j = int(x * 2)
    k = int((theta % 360) / 30)
    return visited[i][j][k] == 1

def set_visited(x, y, theta):
    i = int(y * 2)
    j = int(x * 2)
    k = int((theta % 360) / 30)
    visited[i][j][k] = 1

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)*2 + (y2 - y1)*2)

def is_goal(node, goal):
    distance = euclidean_distance(node.x, node.y, goal[0], goal[1])
    angle_diff = abs(node.theta - goal[2]) % 360
    return distance <= 1.5 and (angle_diff == 0 or angle_diff == 360)

def generate_new_node(node, step_size, angle):
    theta = (node.theta + angle) % 360
    rad = math.radians(theta)
    new_x = node.x + step_size * math.cos(rad)
    new_y = node.y + step_size * math.sin(rad)
    return round(new_x, 1), round(new_y, 1), theta

def backtrack_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path

def a_star(start, goal, step_size):
    start_node = Node(start[0], start[1], start[2], 0, None)
    start_node.cost_to_go = euclidean_distance(start[0], start[1], goal[0], goal[1])
    start_node.total_cost = start_node.cost_to_go
    open_list = []
    heapq.heappush(open_list, start_node)
    explored = []
    parent_map = {}
    while open_list:
        current = heapq.heappop(open_list)
        if is_goal(current, goal):
            print("Goal Reached!")
            return current, explored, parent_map
        if is_visited(current.x, current.y, current.theta):
            continue
        set_visited(current.x, current.y, current.theta)
        explored.append((current.x, current.y))
        for move in get_motion_set(step_size):
            new_x, new_y, new_theta = generate_new_node(current, move[0], move[1])
            if not is_valid(new_x, new_y):
                continue
            if is_visited(new_x, new_y, new_theta):
                continue
            cost_to_come = current.cost_to_come + step_size
            cost_to_go = euclidean_distance(new_x, new_y, goal[0], goal[1])
            new_node = Node(new_x, new_y, new_theta, cost_to_come, current)
            new_node.cost_to_go = cost_to_go
            new_node.total_cost = cost_to_come + cost_to_go
            heapq.heappush(open_list, new_node)
            parent_map[(new_x, new_y)] = (current.x, current.y)
    print("No path found.")
    return None, explored, parent_map

# ===========================
# Visualization
# ===========================
def plot_final_trajectory_with_vectors(grid_width, grid_height, explored_nodes, optimal_path, clearance_mask, obstacle_mask, parent_map):
    fig, ax = plt.subplots(figsize=(9, 5))
    flipped_clearance = cv2.flip(clearance_mask, 0)
    flipped_obstacle = cv2.flip(obstacle_mask, 0)
    img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    img[np.where(flipped_clearance == 255)] = [0, 255, 0]
    img[np.where(flipped_obstacle == 255)] = [255, 0, 0]
    ax.imshow(img)
    for node in explored_nodes:
        if node in parent_map:
            parent = parent_map[node]
            dx = node[0] - parent[0]
            dy = node[1] - parent[1]
            ax.quiver(parent[0], grid_height - parent[1], dx, -dy, color='blue', angles='xy', scale_units='xy', scale=1, width=0.003)
    if optimal_path:
        x_coords, y_coords = zip(*optimal_path)
        ax.plot(x_coords, [grid_height - y for y in y_coords], color='yellow', linewidth=2, label="Optimal Path")
    ax.scatter(start_position[0], grid_height - start_position[1], color='green', s=50, label="Start")
    ax.scatter(goal_position[0], grid_height - goal_position[1], color='red', s=50, label="Goal")
    ax.set_title("Final A* Trajectory with Node Exploration Vectors")
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def visualize_astar(grid_width, grid_height, explored_nodes, optimal_path, clearance_mask, obstacle_mask):
    """
    Generates a video of the A* exploration process.
    """
    # Create a blank white canvas (RGB image)
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    canvas[np.where(clearance_mask == 255)] = [0, 255, 0]  # Green for clearance
    canvas[np.where(obstacle_mask == 255)] = [255, 0, 0]   # Red for obstacles
    
    # Draw start and goal points
    cv2.circle(canvas, start_position, 1, (0, 0, 0), -1)  # Black for start position
    cv2.circle(canvas, goal_position, 1, (0, 0, 0), -1)   # Black for goal position
    
    # Create a directory to save frames
    frame_dir = "astar_frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_index = 0  # Frame counter
    
    # Plot explored nodes and save frames
    for i, node in enumerate(explored_nodes):
        if i % 50 == 0:  # Save every 50th frame to reduce video size
            cv2.circle(canvas, (int(node[0]), int(node[1])), 1, (0, 0, 255), -1)  # Blue for explored nodes
            cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_index:04d}.png"), canvas)
            frame_index += 1
    
    # Plot the optimal path and save frames
    if optimal_path:
        for i, node in enumerate(optimal_path):
            cv2.circle(canvas, (int(node[0]), int(node[1])), 2, (0, 255, 255), -1)  # Yellow for the optimal path
            cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_index:04d}.png"), canvas)
            frame_index += 1
    
    # Convert saved frames into a video file
    video_path = "astar.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for video writing
    video = cv2.VideoWriter(video_path, fourcc, 60, (grid_width, grid_height))  # 60 FPS video
    
    # Read and write frames into the video
    for i in range(frame_index):
        frame = cv2.imread(os.path.join(frame_dir, f"frame_{i:04d}.png"))
        video.write(frame)
    
    # Release the video writer
    video.release()
    
    # Clean up temporary frames
    shutil.rmtree(frame_dir)
    print(f"Visualization saved as {video_path}")
    return video_path

# Main execution
if _name_ == "_main_":
    start_time = time.time()
    # Prompt user for valid start position
    while True:
        start_x = int(input("Enter X coordinate of start position: "))
        start_y = int(input("Enter Y coordinate of start position: "))
        start_orientation = int(input("Enter orientation of start position: "))
        if start_x < 0 or start_y < 0 or start_x >= canvas_width or start_y >= canvas_height:
            print("Start position is outside the canvas. Please enter a new start position.")
            continue
        elif clearance_mask[start_y, start_x] == 200 or clearance_mask[start_y, start_x] == 255:
            print("Start position is inside an obstacle. Please enter a new start position.")
            continue
        else:
            break
    # Prompt user for valid goal position
    while True:
        goal_x = int(input("Enter X coordinate of goal position: "))
        goal_y = int(input("Enter Y coordinate of goal position: "))
        goal_orientation = int(input("Enter orientation of goal position: "))
        if goal_x < 0 or goal_y < 0 or goal_x >= canvas_width or goal_y >= canvas_height:
            print("Goal position is outside the canvas. Please enter a new goal position.")
            continue
        elif clearance_mask[goal_y, goal_x] == 200 or clearance_mask[goal_y, goal_x] == 255:
            print("Goal position is inside an obstacle. Please enter a new goal position.")
            continue
        else:
            break
    start_position = (start_x, start_y)
    goal_position = (goal_x, goal_y)
    start_position = (start_position[0], canvas_height - start_position[1])
    goal_position = (goal_position[0], canvas_height - goal_position[1])
    start = (start_x, canvas_height - start_y, start_orientation)
    goal = (goal_x, canvas_height - goal_y, goal_orientation)
    radius = 1.5
    step_size = 1
    goal_node, explored_nodes, parent_map = a_star(start, goal, step_size)
    if goal_node:
        path = backtrack_path(goal_node)
        visualize_astar(canvas_width, canvas_height, explored_nodes, path, clearance_mask, obstacle_mask)
        plot_final_trajectory_with_vectors(canvas_width, canvas_height, explored_nodes, path, clearance_mask, obstacle_mask, parent_map)
    else:
        print("Path could not be found.")
    print("Execution Time: %.2f seconds" % (time.time() - start_time))