import numpy as np
import torch
import torch.nn as nn
from PIL import ImageChops
import globals

LEN_EVENTS = len(globals.EVENTS) # dont call unnecessary function in already crowded loop

class Agent:
    def __init__(self, model,value_net,vm,learning_every_n_steps=5):
        self.model = model
        self.value_net = value_net
        self.vm = vm
        self.learning_every_n_steps = learning_every_n_steps

        self.optimizer = nn.AdaDelta(parameters=self.model.parameters(), lr=0.5)
        self.loss_func = nn.CrossEntropyLoss()

        self.score = 0 # score since last learning_step, decided by value net, is [0,1]
        self.left_to_learn = self.learning_every_n_steps
        self.event_history = [] # cleared every learning step
        self.last_frame = None
        self.last_event = "" # for info

    def step(self):
        probs = self.model(self.get_game_screen_with_delta())
        # p is the probability distribution set by policy net
        chosen_event = np.random.choice(globals.EVENTS, p=probs[0])
        self.event_history.append((chosen_event,probs))
        self.vm.keyDown(chosen_event)

        self.last_event = chosen_event
        self.left_to_learn -= 1
        if self.left_to_learn <= 0:
            self.learning_step()

    def learning_step(self):
        new_score = self.value_net(self.vm.get_game_screen())
        score_diff = new_score - self.score

        for event,probs in self.event_history:
            self.optimizer.zero_grad()

            event_idx = globals.EVENTS_IDX[event]
            if score_diff >= 0:
                target = np.zeros((1,LEN_EVENTS))
                target[0][event_idx] = 1.0
            else:
                target = np.ones((1,LEN_EVENTS))
                target[0][event_idx] = 0.0

            loss = self.loss_func(probs, target)
            loss.backward()
            self.optimizer.step()

        # end learning step
        self.score = new_score
        self.event_history = []
        self.left_to_learn = self.learning_every_n_steps



    def get_game_screen_with_delta(self): # appends delta of movement as channel to normal game screen data
        curr_frame = self.vm.get_game_screen(return_im = True)
        if not self.last_frame: self.last_frame = curr_frame
        diff_frame = ImageChops.subtract(self.last_frame,curr_frame).convert("L")

        rgb_arr = np.array(curr_frame.getdata()).astype(np.float32)
        dlt_arr = np.array(diff_frame.getdata()).astype(np.float32).reshape(-1, 1)
        res_arr = np.concatenate((rgb_arr,dlt_arr), axis=1)
        res_arr *= (1.0/255.0) # normalize
        res_arr = res_arr.reshape((1, 4, curr_frame.size[1],curr_frame.size[0])) #(bN,C,H,W)

        self.last_frame = curr_frame
        return res_arr
