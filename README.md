
# Surface Classification for Omni Robot

## Objective

Apply machine learning algorithms to the following task – detect surfaces using data from internal sensors of the robot.

Following types of underlying surface given:

1. **Grey** – slippery rubber material
2. **Green** – hard grass-like plastic material
3. **Table** – wooden surface

## Robot

Festo Robotino is used as the data collecting platform. The robot comes with omnidirectional drives and different types of sensors. The platform's schematic is in the picture.
- Training data were collected with Robotino v1.6
- Validation data were collected with Robotino v4

![Appearance and Coordinate system of the robot](https://user-images.githubusercontent.com/35947176/216916408-7a03bd91-a63a-4992-a56d-6c8579ca362e.png)

## Data

Training dataset contains direct and indirect measurements acquired from mobile platform during experiments for each surface type. 

**Direct measurements** are:

- Motor current for each motor
- Encoder ticks for each wheel

**Indirect measurements** (portion) are:

- Wheel speed for each motor
- Axis currents for robot's axes
- Motor voltage and torque for each motor
- etc.

There are 3 diffrerent parameter sets:

1. Motors' currents, total motors' current,  axes' currents, total axes' current
2. Motors' currents, wheels' speeds
3. Motor's currents

Also each set of parameters includes stated speeds for X and Y axes and logical variable that represents Rotational component in robot's motion.

## Models

Since a multi-class classification task is given following algorithms will be used:

- Decision Tree
- Random Forest
- LightGBM
- CatBoost
