
# Surface Classification for Omni Robot

## Objective

Apply machine learning algorithms to the following task – detect surfaces using data from internal sensors of the robot.

Following types of underlying surface given:

1. **Grey** – slippy rubber material
2. **Green** – hard grass-like plastic material
3. **Table** – wooden surface

## Robot

![Appearance and Coordinate system of the robot](https://user-images.githubusercontent.com/35947176/216916408-7a03bd91-a63a-4992-a56d-6c8579ca362e.png)

## Data

Training dataset contains direct and indirect measurements acquired from mobile platform during experiments for each surface type.

**Direct measurements** are:

- Motor current for each motor
- Encoder ticks for each wheel

**Indirect measurements** (portion) are:

- Wheel speed for each motor
- Axis current for robot's axes
- Motor voltage and torque for each motor
- etc.



