import os, random, time, math, sys
import numpy as np
import torch
import model_creator
import globals
import vm_intf
import colored
vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)
model = model_creator.create_score_model()
model.load_state_dict(torch.load(globals.DIR + "/ValueNet.model"))
model.train(False)

BOXES_NUM = 40
COLORS_FG = (2, 3, 9, 1)


MEAN_N = 10
VALS = [0]*MEAN_N

while True:
    time.sleep(0.1)
    vm.refreshScreen()
    with torch.no_grad():
        pred_value = model(torch.Tensor(vm.get_game_screen()))

        if globals.VALUENET_REGRESSOR:
            pred_value = float(pred_value)
        else:
            max_ind = pred_value.max(dim=1)[1]
            label = globals.VALUENET_DIRS[max_ind]
            pred_value = float(label)/100

    VALS.append(pred_value)
    curr_pred_value = pred_value

    if len(VALS) > MEAN_N:
        pred_value = sum(VALS[-MEAN_N:])/MEAN_N # take mean of last MEAN_N

    fg_chosen = math.floor(len(COLORS_FG)*pred_value)
    fg_chosen = colored.fg(COLORS_FG[fg_chosen])
    _s = "[" # %s for blocks
    r = round(pred_value*BOXES_NUM)
    for i in range(r):
        _s += "█"
    while len(_s) < BOXES_NUM+1:
        _s += "-"
    _s += "] => "
    _s += str(round(float(pred_value*100))) + "% | " + str(round(float(curr_pred_value*100))) + "%  "
    print(fg_chosen + _s, end="\r")
