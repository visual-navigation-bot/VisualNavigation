import make
import numpy as np
import ctrl_iLQR_LTA as ctrl

def show_observation(observation):
    print "#####Observation#####"
    print "AGENT INFO:"
    print "    position: ",observation['agent_position']
    print "    velocity: ",observation['agent_velocity']
    print "    goal position: ", observation['agent_goal_position']
    print "PEDESTRIAN INFO:"
    for ped_index in range(len(observation['ped_ID'])):
        print "    ID: ", observation['ped_ID'][ped_index]
        print "    position: ", observation['ped_position'][ped_index]
        print "    velocity: ", observation['ped_velocity'][ped_index]
        print ""
    print "#####################"

env = make.make_environment('LTA_Continuous_ver0_Two_Peds_Walk')

params = {
        }
predicter = make.make_environment('LTA_Continuous_ver0_Two_Peds_Walk')
controller = ctrl.ILQR_LTA_Controller(predicter, params)

env.display()
obs = env.reset()
done = False
while not done:
    action = controller.control(obs)
    obs, reward, done = env.step(action)


