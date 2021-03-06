import os, random, time, math, sys
import numpy as np
import torch
import model_creator
import globals
import vm_intf
import colored
from functools import reduce

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)
model = model_creator.create_score_model()
model.load_state_dict(torch.load(globals.DIR + "/ValueNet.model"))
model.train(False)

BOXES_NUM = 40
COLORS_FG = (2, 3, 9, 1)


MEAN_N = 5
VALS = [0]*MEAN_N


score_abs = 0
while True:
    time.sleep(0.2)
    vm.refreshScreen()
    game_state = vm.get_game_screen()
    with torch.no_grad():
        pred_value = model(torch.Tensor(game_state))

        if globals.VALUENET_REGRESSOR:
            pred_value = float(pred_value)
        else:
            max_ind = pred_value.max(dim=1)[1]
            label = globals.VALUENET_DIRS[max_ind]
            pred_value = float(label)/100

    # img = game_state[0]
    # r = 0
    # g = 1
    # b = 2
    # #r_and_b = np.bitwise_and(img[r]==1.,img[b]==0.)
    # r_and_b = np.bitwise_and(img[r]>0.9,img[g]<0.1)
    # red_x = np.where(r_and_b)[1]
    # if len(red_x) > 0:
    #     score_abs = red_x[0]/96
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
