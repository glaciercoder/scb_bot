# Construct gymnasium env
## Action Space
Action space is continuous, 3 dimensions for torques of three wheels.
Use `gymnasium.spaces.Box` to implement, the bounding of which can be set in `env_params.yaml`
## Obvservation Space
Observation Space is continuous, 10 dimensions which are xyz and orientations(quaternion form) of the robot center, torques from 3 wheels, 

## Troubleshooting
1. Before running RemoteAPI, all child scripts in coppeliasim must be detattched.