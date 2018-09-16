import random
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageChops
import globals

LEN_EVENTS = len(globals.EVENTS) # dont call unnecessary function in already crowded loop
coord_conv_x = np.arange(globals.IMAGE_WIDTH)
coord_conv_x = np.tile(coord_conv_x, (globals.IMAGE_HEIGHT,1)).astype(np.float32)
coord_conv_x *= (2.0/(globals.IMAGE_WIDTH-1))
coord_conv_x -= 1

coord_conv_y = np.arange(globals.IMAGE_HEIGHT)
coord_conv_y = np.tile(coord_conv_y, (globals.IMAGE_WIDTH,1)).astype(np.float32)
coord_conv_y = np.transpose(coord_conv_y)
coord_conv_y *= (2.0/(globals.IMAGE_HEIGHT-1))
coord_conv_y -= 1

coord_conv_layer = np.concatenate((coord_conv_x,coord_conv_y), axis=0)
coord_conv_layer = coord_conv_layer.reshape((1,2, globals.IMAGE_HEIGHT,globals.IMAGE_WIDTH))


class Agent:
    def __init__(self, model,value_net,vm,learning_every_n_steps=20,
                 avg_score_buffer_size=5,
                 past_score_horizon=10,
                 lr=0.1,score_epsilon=0.01,
                 round_score_dp=2):

        self.model = model
        self.value_net = value_net
        self.vm = vm

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()

        self.score = 0 # score updated every step
        self.past_score = 0

        if not past_score_horizon: past_score_horizon = event_buffer_size
        self.past_score_horizon = past_score_horizon # how far in past should you look for reward
        self.score_history = [0] # list of past averaged scores

        # score used to judge agent, is not the actual score, but rather an averaged one.
        # this is done to average out the noise caused by blue critters' movement
        self.avg_score_buffer = [0.0]*avg_score_buffer_size
        self.avg_score_buffer_size = avg_score_buffer_size
        self.round_score_dp = round_score_dp # lower dp to filter out some noise

        # procs learning step every n_steps
        # if n_steps <= 0, procs learning step whenever score improves (score diff > epsilon)
        self.learning_every_n_steps = learning_every_n_steps
        self.left_to_learn = self.learning_every_n_steps # counter variable

        # small_number, considered to gain no reward if score_diff < eps
        self.SCORE_EPSILON = score_epsilon

        # last_frame is used to compute differences in movement for better decision making
        self.last_frame = vm.get_game_screen()

        # information variables (for logging)
        self.last_event = "" # for info
        self.step_counter = 0
        self.positive_step_counter = 0
        self.negative_step_counter = 0


    #NOTE: Learning step and normal step are to be implemented by their children!

    def count_step(self):
        self.step_counter += 1

        self.score_history.append(self.score)
        self.past_score = self.score
        self.score = round(self.get_score(), self.round_score_dp)
        while len(self.score_history) > self.past_score_horizon:
            self.score_history.pop(0)

    def compute_reward(self):
        if abs(self.score - self.score_history[-1]) >= self.SCORE_EPSILON:
            # weighted_rewards = [w*i for i,w in enumerate(self.score_history)]
            # weighted_sum = sum(weighted_rewards)
            # ar_sum = ((self.past_score_horizon+1)/2)*(self.past_score_horizon)
            #
            # reward = weighted_sum/ar_sum
            reward = self.score - self.score_history[-1]
        else:
            reward = -0.1 # bad agent! Explore!

        return reward

    def step(self):
        raise NotImplementedError

    def learning_step(self):
        raise NotImplementedError

    def end_learning_step(self):
        self.left_to_learn = self.learning_every_n_steps

    def get_score(self):
        new_score = self.value_net(torch.tensor(self.vm.get_game_screen()))
        new_score = float(new_score)
        new_score = round(new_score, self.round_score_dp)
        self.avg_score_buffer.append(new_score)
        while len(self.avg_score_buffer) > self.avg_score_buffer_size:
            self.avg_score_buffer.pop(0)
        new_score = sum(self.avg_score_buffer)/self.avg_score_buffer_size
        return new_score



    def get_game_state(self):
        curr_frame = self.vm.get_game_screen()

        # add coordconv (they are created statically at the top of the file!)
        res_arr = np.concatenate((curr_frame,self.last_frame,coord_conv_layer), axis=1)

        self.last_frame = curr_frame
        return torch.tensor(res_arr)

    # DEPRECATED: Wastes too much CPU power which my laptop can't handle :(
    # appends delta of movement as channel to normal game screen data
    # also, appends x and y coordinates as separate channels
    # See Uber AI's paper for more info: https://arxiv.org/pdf/1807.03247.pdf
    def get_game_screen_with_delta(self):
        curr_frame = self.vm.get_game_screen(return_im = True)
        if not self.last_frame: self.last_frame = curr_frame
        diff_frame = ImageChops.subtract(self.last_frame,curr_frame).convert("L")

        rgb_arr = np.array(curr_frame.getdata()).astype(np.float32)
        dlt_arr = np.array(diff_frame.getdata()).astype(np.float32).reshape(-1, 1)
        res_arr = np.concatenate((rgb_arr,dlt_arr), axis=1)
        res_arr *= (2.0/255.0)
        res_arr -= 1 # normalize to [-1,1]

        res_arr = res_arr.reshape((1, 4, curr_frame.size[1],curr_frame.size[0])) #(bN,C,H,W)
        # add coordconv (they are created statically at the top of the file!)
        res_arr = np.concatenate((res_arr,coord_conv_layer), axis=1)

        self.last_frame = curr_frame
        return torch.tensor(res_arr)
