import sys,os,time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model_creator
import vm_intf
import globals

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)
# focus on the game
vm.mouseMove(200,200)
vm.mousePress(1)

# PolicyNet params
model = model_creator.create_model()
optimizer = optim.Adadelta(model.parameters(), lr=0.01)

while True: # main loop
    vm.refreshScreen()
    # drop all events
    for e in EVENTS:
        vm.keyUp(e)

    optimizer.zero_grad()
    im = torch.Tensor(vm.get_game_screen()) # (1,C,H,W) np array for model
    result = model(im)

    np_result = result.detach().numpy()[0]
    max_ind = np.unravel_index(np.argmax(np_result), np_result.shape)[0]
    event = EVENTS[max_ind]
    print("EVENT: " + event)
    vm.keyDown(event)

    time.sleep(globals.TIMESTEP)
