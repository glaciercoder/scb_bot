# Construct gymnasium env
## Action Space
Action space is continuous, 3 dimensions for torques of three wheels.
Use `gymnasium.spaces.Box` to implement, the bounding of which can be set in `env_params.yaml`
## Obvservation Space
Observation Space is continuous, 16 dimensions which are velocity of three wheels, velocity and angular velocity of center of mass, relative xyz and orientations(quaternion form) of the robot center to target pose, sim time.

## Troubleshooting
1. Before running RemoteAPI, all child scripts in coppeliasim must be detattched.