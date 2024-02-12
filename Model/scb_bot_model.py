from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

# The class for robot coppeliasim model, use ZeroMQ API
# Argument: `sim`: Coppeliasim sim object
#           `model_params`: joint_name, etc
class ScbBotModel():
    def __init__(self, sim, model_params:dict) -> None:
        self.sim = sim
        self.params = model_params
        self.jointhds = [None, None, None]
        self.bodyhd = None
        self.graphhd = None
        
        self.joint_torques = np.zeros(3)
        self.target_torques = np.zeros(3)
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        # Retrive joints
        self._get_handlers()
        
    def _get_handlers(self):
        joint_names = self.params['joint_names']
        for i in range(3):
            self.jointhds[i] = self.sim.getObject('/'+joint_names[i])
            print(f'Get joint {joint_names[i]}......')
        self.bodyhd = self.sim.getObject('/' + self.params['body_name'])
        print(f'Get Link {self.params["body_name"]}......')
        self.graphhd = self.sim.getObject('/' + self.params['graph_name'])
        self._set_graph()
        print("Set graph.......")

    def _get_torques(self):
        for i in range(3):
            self.joint_torques[i] = self.sim.getJointForce(self.jointhds[i])

    def _get_pos(self):
        pose = self.sim.getObjectPose(self.bodyhd)
        self.position = np.asarray(pose[:3])
        self.orientation = np.asarray(pose[3:])

    def update_state(self):
        self._get_torques()
        self._get_pos()

    def set_torques(self, target_torques:np.ndarray):
        np.copyto(self.target_torques, target_torques)
        for i in range(3):
            self.sim.setJointForce(self.jointhds[i], target_torques[i])

    def reset_model(self):
        self.sim.setObjectPose(self.bodyhd, [0,0,1,0,0,0,1])
        for i in range(3):
            self.sim.setJointForce(self.jointhds[i], 0)

    def _set_graph(self):
        objectTorque0 = self.sim.addGraphStream(self.graphhd, 'wheel 0 torque', 'N.m', 1)
        self.sim.setGraphStreamValue(self.graphhd, objectTorque0, self.joint_torques[0])

    

    