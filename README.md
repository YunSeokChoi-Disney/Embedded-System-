

<Midterm Project>

This project focused on enabling an autonomous Wave Rover to perceive its surroundings and execute specific driving missions using YOLO-based object detection and PID control.

Task 1: Traffic Light Recognition & Intersection Navigation

*Mission: Detect Red/Green traffic lights at intersections and control the vehicle's movement accordingly.

*Strategy: Implemented a real-time detection logic where the rover stops if a Red light's bounding box exceeds a certain area (indicating proximity) and resumes driving when the signal changes.

Task 2: Sign-based Speed & Stop Control

*Mission: Respond to 'Pedestrian' (Slow down) and 'Stop' signs located on the roadside.

*Strategy: Pulsed PWM (Pulse Width Modulation) to decrease motor output upon detecting a Pedestrian sign. For 'Stop' signs, a timer-based logic was applied to ensure the vehicle remained stationary for a set duration before proceeding.

Task 3: Object Detection & Avoidance Maneuver (Core Task)

*Mission:

*Strategy: Developed a 3-step avoidance algorithm

(1st step) Avoiding: Steer away from the center lane upon detection.

(2nd step) Straight: Maintain a parallel path to bypass the obstacle.

(3rd step) Recovery: Return to the original lane and realign with the center line using PID control.

Task 4: Complex Scenario Integration

*Mission: Navigate intersections by simultaneously processing traffic lights and directional signs (Left/Right/Straight).

*Strategy: Established a decision-making hierarchy to prioritize signals. For instance, the system was programmed to prioritize the Red light signal over any directional signs to ensure safety and compliance.

[Primary Problem Solving & Technical Insights]

Dataset: Created a robust training set with over 27,000 frames to ensure high detection accuracy across various lighting conditions.

Optimization: To resolve oscillation issues during lane-keeping, the PID integral gain (Ki) was fine-tuned (from 0.1 to 0.095), significantly improving driving stability.

Note: For detailed technical documentation, including source code and experimental data, please refer to the attached Mid-term/Final Report and Presentation slides in this repository.

<Final Project> 

The final phase of this project focused on enhancing the vehicle's decision-making logic and stabilizing its driving performance based on feedback from the mid-term evaluation.

1. Solving Key Challenges (Feedback Loop)
   
*Stabilizing Control: Addressed the oscillation issues during lane-keeping by fine-tuning the PID parameters, resulting in much smoother and more reliable path-following.

*Enhanced Perception: Expanded the dataset and refined the YOLO model to ensure more consistent detection of various vehicle types and traffic signals under different lighting conditions.

2. Advanced Mission: Dynamic Obstacle Response

*The "Wait or Overtake" Logic: Unlike the mid-term project which only dealt with static objects, the final system can now respond to moving vehicles.

*Decision Making: The rover is programmed to wait for a specific duration (e.g., 5 seconds) if a vehicle is detected ahead. If the obstacle remains stationary, the rover automatically initiates the 3-step avoidance maneuver (Avoiding → Straight → Recovery).
