from numpy import cos, sin, deg2rad, round, sqrt
import time
import heapq
import cv2
import numpy as np

# Define the Node class to store node information
class Node:
    """Class to store the node information."""
    def __init__(self, coords, cost, parent=None, heuristic=0):
        self.coords = coords  # Node's position (x, y, theta)
        self.x = coords[0]   # x-coordinate
        self.y = coords[1]   # y-coordinate
        self.orientation = coords[2]  # Orientation (theta)
        self.cost = cost  # Cost to reach this node
        self.parent = parent  # Parent node for backtracking
        self.heuristic = heuristic  # Heuristic cost to goal

    def __lt__(self, other):
        """Comparison operator for priority queue."""
        return self.cost + self.heuristic < other.cost + other.heuristic


# Obstacle-checking functions
def is_in_obstacle_1(x, y, x0=390+60, y0=145, width=10, height=40, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    # Vertical bar of "1"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    return False


def is_in_obstacle_6_second(x, y, x0=340+60, y0=145, large_radius=20, medium_radius=14,
                           small_radius=8, hole_radius=5, thickness=3):
    """Defines the second '6' using arcs with a small hole in the center."""
    # Top arc of "6"
    top_x = x0 + medium_radius // 2
    top_y = y0 - large_radius * 1.5
    inside_top = ((x - top_x) ** 2 + (y - top_y) ** 2) <= small_radius ** 2 and x >= top_x

    # Middle arc of "6"
    mid_x = x0 + large_radius - thickness
    mid_y = y0 - large_radius
    inside_middle = ((x - mid_x) ** 2 + (y - mid_y) ** 2) <= large_radius ** 2 and x <= mid_x

    # Bottom arc of "6"
    bottom_x = x0 + medium_radius
    bottom_y = y0 - medium_radius
    inside_bottom = ((x - bottom_x) ** 2 + (y - bottom_y) ** 2) <= medium_radius ** 2 and x >= bottom_x + thickness

    # Hole in the middle of "6"
    hole_x = x0 + medium_radius
    hole_y = y0 - medium_radius
    inside_hole = ((x - hole_x) ** 2 + (y - hole_y) ** 2) <= hole_radius ** 2

    # Return True if the point is inside "6" but not in the hole
    if (inside_top or inside_middle or inside_bottom) and not inside_hole:
        return True
    return False


def is_in_obstacle_6_first(x, y, x0=285+60, y0=145, large_radius=20, medium_radius=14,
                          small_radius=8, hole_radius=5, thickness=3):
    """Defines the first '6' using arcs with a small hole in the center."""
    # Similar logic to `is_in_obstacle_6_second`
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


def is_in_obstacle_M(x, y, x0=230+60, y0=145, width=30, height=40, thickness=3):
    """Defines the letter 'M' using vertical bars and diagonal slopes."""
    # Check if the width is too small to define "M"
    if width <= 2 * thickness:
        return False

    # Left vertical bar of "M"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True

    # Right vertical bar of "M"
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True

    # Left diagonal slope of "M"
    slope_left = (height / 2) / (width / 2 - thickness)
    y_expected_left = slope_left * (x - x0 - thickness) + (y0 - height)
    if x0 + thickness <= x <= x0 + width / 2 and y_expected_left <= y <= y_expected_left + 10:
        return True

    # Right diagonal slope of "M"
    slope_right = (-height / 2) / (width / 2 - thickness)
    y_expected_right = slope_right * (x - (x0 + width / 2)) + (y0 - height / 2)
    if x0 + width / 2 <= x <= x0 + width - thickness and y_expected_right <= y <= y_expected_right + 10:
        return True

    return False


def is_in_obstacle_P(x, y, x0=200+60, y0=145, width=20, height=40, thickness=3):
    """Defines the letter 'P' using a vertical bar and a semi-circular top."""
    # Vertical bar of "P"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True

    # Semi-circular top of "P"
    curve_x_center = x0 + thickness
    curve_y_center = y0 - (3 * height / 4)
    curve_radius = height / 4
    if ((x - curve_x_center) ** 2 + (y - curve_y_center) ** 2) <= curve_radius ** 2 and y < y0 - height // 2 and x > x0 + thickness:
        return True

    return False


def is_in_obstacle_N(x, y, x0=160+60, y0=145, width=20, height=40, thickness=3):
    """Defines the letter 'N' using two vertical bars and a diagonal."""
    # Check if the width is too small to define "N"
    if width <= 2 * thickness:
        return False

    # Left vertical bar of "N"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True

    # Right vertical bar of "N"
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True

    # Diagonal of "N"
    slope = (2/3)*height / (width - 2 * thickness)
    y_expected = (slope * (x - (x0 + thickness))) + (y0 - height)
    if x0 + thickness <= x <= x0 + width - thickness and y_expected <= y <= y_expected + height/3:
        return True

    return False


def is_in_obstacle_E(x, y, x0=120+60, y0=145, width=20, height=40, mid_width=15, thickness=3):
    """Defines the letter 'E' using three horizontal bars and a vertical bar."""
    # Vertical bar of "E"
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True

    # Top horizontal bar of "E"
    if x0 <= x <= x0 + width and y0 - height <= y <= y0 - height + thickness:
        return True

    # Middle horizontal bar of "E"
    if x0 <= x <= x0 + mid_width and y0 - height // 2 - thickness // 2 <= y <= y0 - height // 2 + thickness // 2:
        return True

    # Bottom horizontal bar of "E"
    if x0 <= x <= x0 + width and y0 - thickness <= y <= y0:
        return True

    return False


def generate_map():
    """Creates the map with obstacles and clearance zones."""
    # Initialize an empty canvas
    grid = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Set the offset for clearance and robot radius
    offset = clearance + robot_radius

    # Traverse through each pixel in the canvas
    for i in range(canvas_height):
        for j in range(canvas_width):
            # Check if the pixel is inside any obstacle
            if (is_in_obstacle_1(j, i) or
                is_in_obstacle_6_second(j, i) or
                is_in_obstacle_6_first(j, i) or
                is_in_obstacle_M(j, i) or
                is_in_obstacle_P(j, i) or
                is_in_obstacle_N(j, i) or
                is_in_obstacle_E(j, i)):
                grid[i][j] = 1  # Mark as obstacle

            # Add clearance around obstacles
            elif (is_in_obstacle_1(j, i, thickness=offset) or
                  is_in_obstacle_6_second(j, i, thickness=offset) or
                  is_in_obstacle_6_first(j, i, thickness=offset) or
                  is_in_obstacle_M(j, i, thickness=offset) or
                  is_in_obstacle_P(j, i, thickness=offset) or
                  is_in_obstacle_N(j, i, thickness=offset) or
                  is_in_obstacle_E(j, i, thickness=offset)):
                grid[i][j] = 1  # Mark as clearance zone

            # Define borders (edges of the canvas)
            if (j < offset or j >= canvas_width - offset or
                i < offset or i >= canvas_height - offset):
                grid[i][j] = 1  # Mark as border

    return grid


def is_valid_point(x, y, grid):
    """Checks whether the given coordinates are valid (not on an obstacle)."""
    # Return False if the point is on an obstacle
    if grid[y][x] == 1:
        return False
    return True


def create_animation_grid():
    """Creates the animation grid with obstacles, clearance zones, and borders."""
    # Create a white background canvas
    image = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Set the offset for clearance and robot radius
    offset = clearance + robot_radius

    # Define colors for objects
    clearance_color = (255, 255, 0)  # Cyan for clearance zones
    border_color = (200, 200, 200)   # Light gray for borders
    obstacle_color = (255, 0, 255)   # Magenta for obstacles

    # Traverse through each pixel in the canvas
    for i in range(canvas_height):
        for j in range(canvas_width):
            # Represent actual obstacles (highest priority)
            if (is_in_obstacle_1(j, i) or
                is_in_obstacle_6_second(j, i) or
                is_in_obstacle_6_first(j, i) or
                is_in_obstacle_M(j, i) or
                is_in_obstacle_P(j, i) or
                is_in_obstacle_N(j, i) or
                is_in_obstacle_E(j, i)):
                image[i][j] = obstacle_color  # Magenta for obstacles

            # Represent clearance zone (second priority)
            elif (is_in_obstacle_1(j, i, thickness=clearance) or
                  is_in_obstacle_6_second(j, i, thickness=clearance) or
                  is_in_obstacle_6_first(j, i, thickness=clearance) or
                  is_in_obstacle_M(j, i, thickness=clearance) or
                  is_in_obstacle_P(j, i, thickness=clearance) or
                  is_in_obstacle_N(j, i, thickness=clearance) or
                  is_in_obstacle_E(j, i, thickness=clearance)):
                image[i][j] = clearance_color  # Cyan for clearance zone

            # Represent borders (lowest priority)
            elif (j < offset or j >= canvas_width - offset or
                  i < offset or i >= canvas_height - offset):
                image[i][j] = border_color  # Light gray for borders

    return image


def round_value(number):
    """Rounds the given number to nearest 0.5."""
    return np.around(number * 2.0) / 2.0


def move_positive_60(coords, cost):
    """Moves the robot forward at an angle of positive 60 degrees."""
    x, y, theta = coords
    theta += 60  # Update orientation
    x = x + (step_size * cos(deg2rad(theta)))  # Update x-coordinate
    y = y + (step_size * sin(deg2rad(theta)))  # Update y-coordinate
    cost += 1  # Increment cost
    return [[x, y, theta], cost]


def move_positive_30(coords, cost):
    """Moves the robot forward at an angle of positive 30 degrees."""
    x, y, theta = coords
    theta += 30  # Update orientation
    x = x + (step_size * cos(deg2rad(theta)))  # Update x-coordinate
    y = y + (step_size * sin(deg2rad(theta)))  # Update y-coordinate
    cost += 1  # Increment cost
    return [[x, y, theta], cost]


def move_straight(coords, cost):
    """Moves the robot straight forward."""
    x, y, theta = coords
    x = x + (step_size * cos(deg2rad(theta)))  # Update x-coordinate
    y = y + (step_size * sin(deg2rad(theta)))  # Update y-coordinate
    cost += 1  # Increment cost
    return [[x, y, theta], cost]


def move_negative_30(coords, cost):
    """Moves the robot forward at an angle of negative 30 degrees."""
    x, y, theta = coords
    theta -= 30  # Update orientation
    x = x + (step_size * cos(deg2rad(theta)))  # Update x-coordinate
    y = y + (step_size * sin(deg2rad(theta)))  # Update y-coordinate
    cost += 1  # Increment cost
    return [[x, y, theta], cost]


def move_negative_60(coords, cost):
    """Moves the robot forward at an angle of negative 60 degrees."""
    x, y, theta = coords
    theta -= 60  # Update orientation
    x = x + (step_size * cos(deg2rad(theta)))  # Update x-coordinate
    y = y + (step_size * sin(deg2rad(theta)))  # Update y-coordinate
    cost += 1  # Increment cost
    return [[x, y, theta], cost]


def is_valid_child(child):
    """Checks whether the child node is valid (not outside the canvas or on an obstacle)."""
    x, y, _ = child[0]
    # Check if the child node is within the canvas boundaries
    if 0 <= x <= canvas.shape[1] - 1 and 0 <= y <= canvas.shape[0] - 1:
        # Check if the child node is not on an obstacle
        if canvas[int(round(y))][int(round(x))] == 0:
            return True
    return False


def generate_successors(node):
    """Generates all valid successor nodes of a given node."""
    x, y, theta = node.coords
    cost = node.cost
    successors = []  # List to store valid child nodes

    # Generate child nodes for different movements
    p60 = move_positive_60([x, y, theta], cost)
    if is_valid_child(p60):
        successors.append([[round_value(p60[0][0]), round_value(p60[0][1]), p60[0][2]], p60[1]])

    p30 = move_positive_30([x, y, theta], cost)
    if is_valid_child(p30):
        successors.append([[round_value(p30[0][0]), round_value(p30[0][1]), p30[0][2]], p30[1]])

    frd = move_straight([x, y, theta], cost)
    if is_valid_child(frd):
        successors.append([[round_value(frd[0][0]), round_value(frd[0][1]), frd[0][2]], frd[1]])

    n30 = move_negative_30([x, y, theta], cost)
    if is_valid_child(n30):
        successors.append([[round_value(n30[0][0]), round_value(n30[0][1]), n30[0][2]], n30[1]])

    n60 = move_negative_60([x, y, theta], cost)
    if is_valid_child(n60):
        successors.append([[round_value(n60[0][0]), round_value(n60[0][1]), n60[0][2]], n60[1]])

    return successors


def check_visited(x, y, theta):
    """Checks duplicate nodes."""
    xn = int(round_value(x) / visited_threshold)
    yn = int(round_value(y) / visited_threshold)
    if theta <= 0:
        thetan = int(360 / (360 - theta)) - 1
    else:
        thetan = int(360 / theta) - 1
    if visited_mat[yn][xn][thetan] == 0:  # If not visited, mark as visited
        visited_mat[yn][xn][thetan] = 1
        return True
    return False


def astar_search(start_node, goal_node):
    """Performs A* search to find the shortest path."""
    # Create the animation canvas
    animation_grid = create_animation_grid()
    animation_frames = []

    # Initialize data structures for A* algorithm
    node_graph = {}
    open_list = {}
    closed_list = {}
    queue = []

    # Add the start node to the open list and priority queue
    open_list[str([start_node.x, start_node.y])] = start_node
    heapq.heappush(queue, [start_node.cost, start_node])

    i = 0  # Counter for animation frames
    while len(queue) != 0:
        # Fetch the node with the lowest cost from the priority queue
        fetched_ele = heapq.heappop(queue)
        current_node = fetched_ele[1]

        # Add the current node to the node graph
        node_graph[str([current_node.x, current_node.y])] = current_node

        # Draw the current node on the animation canvas
        cv2.circle(animation_grid, (int(current_node.x), int(current_node.y)), 1, (0, 0, 255), -1)

        # Check if the goal has been reached
        if sqrt((current_node.x - goal_node.x) ** 2 + (current_node.y - goal_node.y) ** 2) < goal_threshold:
            goal_node.parent = current_node.parent
            goal_node.cost = current_node.cost
            print("# Found the goal node")
            break

        # Add the current node to the closed list
        if str([current_node.x, current_node.y]) in closed_list:
            continue
        else:
            closed_list[str([current_node.x, current_node.y])] = current_node

        # Remove the current node from the open list
        del open_list[str([current_node.x, current_node.y])]

        # Generate children nodes for the current node
        child_list = generate_successors(current_node)
        for child in child_list:
            child_x, child_y, child_theta = child[0]
            child_cost = child[1]

            # Draw the child node on the animation canvas
            cv2.circle(animation_grid, (int(child_x), int(child_y)), 1, (0, 255, 0), -1)

            # Save animation frames periodically
            if i % 1000 == 0:
                animation_frames.append(animation_grid.copy())

            # Skip if the child node is already in the closed list
            if str([child_x, child_y]) in closed_list:
                continue

            # Calculate the heuristic cost to the goal
            child_heuristic = sqrt((goal_node.x - child_x) ** 2 + (goal_node.y - child_y) ** 2)

            # Create a new child node
            child_node = Node([child_x, child_y, child_theta], child_cost, current_node, child_heuristic)

            # Check if the child node has already been visited
            if check_visited(child_x, child_y, child_theta):
                if str([child_x, child_y]) in open_list:
                    # Update the cost if a cheaper path is found
                    if child_node.cost < open_list[str([child_x, child_y])].cost:
                        open_list[str([child_x, child_y])].cost = child_cost
                        open_list[str([child_x, child_y])].parent = current_node
                else:
                    # Add the child node to the open list and priority queue
                    open_list[str([child_x, child_y])] = child_node
                    heapq.heappush(queue, [(child_cost + child_heuristic), child_node])

        i += 1  # Increment the frame counter

    return node_graph, animation_grid, animation_frames


def backtrack_path(node_graph, goal_node):
    """Backtracks to find the path from start to goal."""
    path = []
    path.append([int(goal_node.x), int(goal_node.y)])
    parent = list(node_graph.items())[-1][1].parent
    while parent:
        path.append([int(parent.x), int(parent.y)])
        parent = parent.parent
    return path


def save_video(frames):
    """Saves the exploration animation as a video."""
    print("#" * 80)
    print("Generating the video file.")
    video = cv2.VideoWriter('path_planning.avi', cv2.VideoWriter_fourcc(*'XVID'), 50, (600, 250))
    for frame in frames:
        video.write(frame)
        cv2.imshow("Exploration", frame)

        # cv2.waitKey(1)
        # Wait for 1 ms between frames and check if the user presses 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    video.release()
    print("Video file generated successfully.")
    print("#" * 80)


def main():
    """Main function to run the program."""
    global canvas, canvas_height, canvas_width
    global step_size, clearance, robot_radius
    global goal_threshold, visited_mat, visited_threshold

    # Initialize global parameters
    canvas_height, canvas_width = 250, 600
    visited_threshold = 0.5
    visited_mat = np.zeros((500, 1200, 12), dtype=np.uint8)
    goal_threshold = 1.5

    loop = True
    while loop:
            # Manual input of parameters
            start_x = int(input("\nEnter x coordinate of Start Point: "))
            start_y = int(input("Enter y coordinate of Start Point: "))
            start_theta = int(input("Enter theta coordinate of Start Point: "))
            if start_theta % 30 != 0:
                print("\nTheta should be a multiple of 30")
                continue
            goal_x = int(input("Enter x coordinate of Goal Point: "))
            goal_y = int(input("Enter y coordinate of Goal Point: "))
            goal_theta = int(input("Enter theta coordinate of Goal Point: "))
            if goal_theta % 30 != 0:
                print("\nTheta should be a multiple of 30")
                continue
            clearance = int(input("Enter the clearance: "))
            if clearance != 5:
                print("\nThe acceptable clearance is 5 as per the instructions")
                continue
            robot_radius = int(input("Enter the robot radius: "))
            if robot_radius != 5:
                print("\nThe acceptable robot radius is 5 as per the instructions")
                continue
            step_size = int(float(input("Enter the step size: ")))
            if step_size not in range(1, 11):
                print("\nThe acceptable step size is 1 <= step_size <= 10")
                continue

            # Create the canvas with obstacles
            canvas = generate_map()

            # Check if the start and goal points are valid
            if is_valid_point(start_x, 250 - start_y, canvas):
                if is_valid_point(goal_x, 250 - goal_y, canvas):
                    # Define the start and goal nodes
                    start_node = Node([start_x, 250 - start_y, start_theta], 0)
                    goal_node = Node([goal_x, 250 - goal_y, goal_theta], 0)
                    print("\nFinding the goal node...")
                    start_time = time.time()
                    node_graph, animation_grid, animation_frames = astar_search(start_node, goal_node)
                    final_path = backtrack_path(node_graph, goal_node)
                    end_time = time.time()
                    print("\nThe output was processed in ", end_time - start_time, " seconds.")
                    print("#" * 80)
                    for point in final_path[::-1]:
                        cv2.circle(animation_grid, point, 1, (255, 0, 0), -1)
                        animation_frames.append(animation_grid.copy())
                    break
            else:
                print("\nThe start point or goal point is on the obstacle")
                continue
    save_video(animation_frames)


if __name__ == "__main__":
    main()