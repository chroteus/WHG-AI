import sys,os,time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model_creator
import vm_intf
import globals
from agent import Agent

# tensorboard logging
from tensorboard_logger import configure, log_value
timestamp = str(round(time.time()))[-6:]
configure(globals.DIR + "/runs/policy_net_"+timestamp, flush_secs=5)

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)
# focus on the game
vm.mouseMove(200,200)
vm.mousePress(1)

# create random policy net
policy_net = model_creator.create_model()
# load value_net
value_net = model_creator.create_score_model()
value_net.load_state_dict(torch.load(globals.DIR + "/ValueNet.model"))
value_net.train(False) # disable dropout
# init agent
agent = Agent(model=policy_net, value_net=value_net, vm=vm,
              learning_every_n_steps=-1,
              event_buffer_size=10,
              lr=0.1,
              chance_coeff=1,
              score_epsilon=0.15,
              past_score_horizon=10,
              avg_score_buffer_size=10,
              round_score_dp=2, # 2 dp
              penalize_if_repeats=True)
print("=== Loaded everything! ===")

SAVED_EVERY_STEPS = 500 # saves progress every ~5 minutes
latest_score = 0
# last_100_scores = [0]*100
max_score = 0
while True:
    vm.refreshScreen()
    agent.step()
    latest_score = agent.score
    # last_100_scores.append(latest_score)
    # while len(last_100_scores) > 100:
    #     last_100_scores.pop(0)

    if latest_score > max_score:
        max_score = latest_score

    # end step
    print(agent.last_event)
    print("Score: " + str(round(agent.score, 5)))
    log_value("Score", latest_score, agent.step_counter)
    log_value("Max Score", max_score, agent.step_counter)
    log_value("Steps Pos.", agent.positive_step_counter, agent.step_counter)
    log_value("Steps Neg.", agent.negative_step_counter, agent.step_counter)

    if agent.step_counter % SAVED_EVERY_STEPS == 0: # save model
        torch.save(policy_net.state_dict(), globals.DIR + "/models/PolicyNet_"+str(agent.step_counter))

    time.sleep(globals.TIMESTEP)
    for e in globals.EVENTS:
        vm.keyUp(e)
