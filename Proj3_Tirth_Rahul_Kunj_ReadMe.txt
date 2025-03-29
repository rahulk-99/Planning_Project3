# ENPM 661 – Planning for Autonomous Robots (Spring 2023)

Project 3, Phase 1  
**Implementation of A* Algorithm for a Mobile Robot**

---

## Team Members

- Name: Tirth Sadaria  
  UID: 121322876  

- Name: Kunj Golwala  
  UID: 121118271 
 
- Name: Rahul Kumar  
  UID: 121161543 

---

## GitHub Link
[GitHub Repository Link](https://github.com/rahulk-99/Planning_Project3.git)



## About the Code
This project implements an A* path planning algorithm for a mobile robot. Specifically, it:

1. Generates an Obstacle Space  
   - Letters/Numbers as obstacles: *E,N,P,M,6,6,1*  
   - Borders and clearance are computed and marked.
2. Possible Movements 
   - The robot can move in 5 directions: +60°, +30°, 0°, -30°, -60° (relative to its current orientation).
3. A* Search
   - Uses a priority queue (heapq) to explore nodes by cost + heuristic.
   - Maintains a visited array to avoid re-processing nodes.
4. Backtracking  
   - After reaching the goal, it retraces parents to generate the shortest path.
5. Visualization  
   - Uses OpenCV (cv2) frames to display obstacles, expansions, and the final path.
   - A video (.mp4) is saved at the end with the animation.

---

## Dependencies

- Python 3.x  
- IDE or Notebook  
  - Code was tested in [Google Colab], but can run in any environment with the listed libraries installed.
- Libraries 
  - numpy
  - heapq (built-in)
  - time (built-in)
  - math (or direct from numpy trig)
  - cv2 (OpenCV for Python)
  - "google.colab.patches" (for `cv2_imshow` in Colab)

Ensure these are installed (e.g., pip install opencv-python numpy).

---

## Instructions to Run the Code

1. Open the Code File  
   - File name might be "Proj3_Tirth_Rahul_Kunj.py" or whichever file you have.


2. Set Parameters 
   - The code can run in two modes:
     - Preset mode: Hard-coded clearance, robot radius, etc.
     - Custom mode: Prompts you for:
       - Clearance
       - Robot Radius
       - Step Size
       - Start coordinates (x, y, theta)
       - Goal coordinates (x, y, theta)
   - Theta should be multiple of 30 degrees.

3. Watch the Console Prompts  
   - Enter numeric values as prompted (e.g., `5` for clearance, `5` for robot radius, etc.).
   - Example:
     - Start point (x, y, theta) = `20, 20, 60`
     - Goal point (x, y, theta) = `400, 20, 30`
     - Clearance = `5`, Robot Radius = `5`, Step size = `1`

5. Output  
   - The code will search for the path and generate frames of the exploration.
   - At the end, it saves `path_planning.mp4` with the visualization.
   
---

## Understanding the Visualization

- Obstacle Space  
  - Obstacles are drawn in magenta (or red/blue, depending on the color definitions).
  - Clearance is drawn in cyan or yellow.
  - Borders are shown in light gray, etc.

- Exploration 
  - Nodes expanded are drawn in one color (green).
  - The final path from start to goal is usually highlighted in another color (red).
  - Circles represent visited nodes, and an animation frames the incremental search.

- Final Path  
  - Shown in a distinct color (blue).
  - Optionally, the start node in purple and goal node in green.

---

## Notes

1. Runtime depends on step size, obstacle density, and heuristics.  
2. Theta multiples of 30° ensures consistent orientation checks.  
3. Code can be modified to incorporate more advanced heuristics, e.g. Weighted A*, or BFS if heuristic is set to zero.

---

## Acknowledgments

- University of Maryland – ENPM 661  
- Libraries: OpenCV, NumPy  

Feel free to raise issues in the repository if you encounter problems or have questions. Good luck and have fun exploring A* path planning!
