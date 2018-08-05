import sys,os,time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model_creator
import vm_intf
import globals
from agent import Agent

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
agent = Agent(model=policy_net, value_net=value_net, vm=vm, learning_every_n_steps=5)
print("=== Loaded everything! ===")

step_counter = 0
latest_score = 0
max_score = 0
while True:
    vm.refreshScreen()
    agent.step()
    latest_score = agent.score
    if latest_score > max_score:
        max_score = latest_score

    for e in globals.EVENTS:
        vm.keyUp(e)

    step_counter += 1
    time.sleep(globals.TIMESTEP)
# while True: # main loop
#     vm.refreshScreen()
#     # drop all events
#     for e in EVENTS:
#         vm.keyUp(e)
#
#     optimizer.zero_grad()
#     im = torch.Tensor(vm.get_game_screen()) # (1,C,H,W) np array for model
#     result = model(im)
#
#     np_result = result.detach().numpy()[0]
#     max_ind = np.unravel_index(np.argmax(np_result), np_result.shape)[0]
#     event = EVENTS[max_ind]
#     print("EVENT: " + event)
#     vm.keyDown(event)
#
#     time.sleep(globals.TIMESTEP)
