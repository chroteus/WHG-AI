import random
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageChops
import globals

LEN_EVENTS = len(globals.EVENTS) # dont call unnecessary function in already crowded loop

class Agent:
    def __init__(self, model,value_net,vm,learning_every_n_steps=5,avg_score_buffer_size=5,
                 lr=0.5,score_epsilon=0.001,chance_coeff=1,
                 event_buffer_size=50, past_score_horizon=20,
                 penalize_if_repeats=True, round_score_dp=12):
        if chance_coeff < 0.3 and disable_negative_steps:
            print("Disabling negative steps not recommended with low chance coefficient")

        self.model = model
        self.value_net = value_net
        self.vm = vm

        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()

        # Whether to just choose max prob. event (0.0) or pick from probability
        # distribution produced by PolicyNet (1.0).
        # It's a float, so you can control how much exploring should be done by agent.
        self.chance_coeff = chance_coeff

        self.score = 0 # score updated every step
        self.past_score = 0
        self.past_score_horizon = past_score_horizon # how far in past should you look
        self.score_history = [] # list of past averaged scores

        # Event history != Event buffer
        # buffer is used for training and has Variables with event probabilities
        # history just stores strings for easier computation
        self.event_history = []
        self.penalize_if_repeats = penalize_if_repeats

        # score used to judge agent, is not the actual score, but rather an averaged one.
        # this is done to average out the noise caused by blue critters' movement
        self.avg_score_buffer = [0.5]*avg_score_buffer_size
        self.avg_score_buffer_size = avg_score_buffer_size
        self.round_score_dp = round_score_dp # lower dp to filter out some noise

        # procs learning step every n_steps
        # if n_steps <= 0, procs learning step whenever score improves (score diff > epsilon)
        self.learning_every_n_steps = learning_every_n_steps
        self.left_to_learn = self.learning_every_n_steps # counter variable

        self.SCORE_EPSILON = score_epsilon # small_number, considered to gain no reward if score_diff <

        # maximum number of events in past store in memory (for training)
        # event_buffer_size only matters when learning step is not constant,
        # to prevent memory overusage
        self.event_buffer_size = event_buffer_size
        self.event_buffer = [] # cleared every learning step or when len(event_buffer) > event_buffer_size

        # last_frame is used to compute differences in movement for better decision making
        self.last_frame = None

        self.last_event = "" # for info
        self.step_counter = 0
        self.positive_step_counter = 0
        self.negative_step_counter = 0

    def step(self):
        self.step_counter += 1

        probs = self.model(self.get_game_screen_with_delta())
        # p is the probability distribution set by policy net
        probs_list = probs.detach().numpy()[0]
        if random.random() > self.chance_coeff:
            chosen_idx = np.argmax(probs_list)
            chosen_event = globals.EVENTS[chosen_idx]
        else:

            chosen_event = np.random.choice(globals.EVENTS, p=probs_list)

        self.event_buffer.append((chosen_event,probs))
        self.vm.keyDown(chosen_event)

        # penalize if agent is repeating itself
        if self.penalize_if_repeats:
            self.event_history.append(chosen_event)
            while len(self.event_history) > self.event_buffer_size:
                self.event_history.pop(0)

            if len(self.event_history) >= self.event_buffer_size \
            and len(set(self.event_history)) == 1: # if all elements are same
                self.negative_step_for_repeating()



        self.score_history.append(self.score)
        self.score = self.get_score()
        while len(self.score_history) > self.past_score_horizon:
            self.score_history.pop(0)
        self.past_score = self.score_history[0]


        self.last_event = chosen_event

        if self.learning_every_n_steps > 0:
            # periodic learning
            self.left_to_learn -= 1
            if self.left_to_learn <= 0:
                self.learning_step()
        else:
            # when score > eps
            while len(self.event_buffer) > self.event_buffer_size:
                self.event_buffer.pop(0)

            score_diff = self.score - self.past_score
            if score_diff >= self.SCORE_EPSILON:
                self.learning_step()

    def learning_step(self):
        score_diff = self.score - self.past_score

        for event,probs in self.event_buffer:
            self.optimizer.zero_grad()

            # do positive rewards only
            event_idx = globals.EVENTS_IDX[event]
            try:
                if score_diff >= self.SCORE_EPSILON:
                    target = np.zeros((1,LEN_EVENTS))
                    target[0][event_idx] = 1
                    self.positive_step_counter += 1
                    print("POSITIVE STEP +++")

                target = torch.tensor(target.astype(np.float32))
                loss = self.loss_func(probs, target)
                loss.backward()
                self.optimizer.step()

            except UnboundLocalError: # no score > eps
                pass

        # end learning step
        self.clear_event_past()
        self.left_to_learn = self.learning_every_n_steps

    def negative_step_for_repeating(self): # used for penalizing if agent repeats itself
        #event,probs = self.event_buffer[0] # just get one datapoint and discard rest, don't want to overtrain
        for event,probs in self.event_buffer:
            self.optimizer.zero_grad()
            event_idx = globals.EVENTS_IDX[event]
            # probability distribution where event we're penalizing is 0
            target = np.ones((1,LEN_EVENTS))*(1/LEN_EVENTS-1)
            target[0][event_idx] = 0
            target = torch.Tensor(target.astype(np.float32))

            loss = self.loss_func(probs, target)
            loss.backward()
            self.optimizer.step()

            self.negative_step_counter += 1
            print("NEGATIVE STEP ---")
            #self.event_buffer.pop(0) # prevents calling backward on same buffer again7
            self.clear_event_past()

    def clear_event_past(self):
        self.event_buffer = []
        self.event_history = []

    def get_score(self):
        new_score = self.value_net(torch.tensor(self.vm.get_game_screen()))
        new_score = float(new_score)
        new_score = round(new_score, self.round_score_dp)
        self.avg_score_buffer.append(new_score)
        while len(self.avg_score_buffer) > self.avg_score_buffer_size:
            self.avg_score_buffer.pop(0)
        new_score = sum(self.avg_score_buffer)/self.avg_score_buffer_size
        return new_score

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
        return torch.tensor(res_arr)
