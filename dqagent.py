import random,math
import numpy as np
import torch
from agent import Agent
import globals
from memory import MemoryBuffer


class DQAgent(Agent):
    def __init__(self, target_model=None, # must be the same as normal model
                       update_target_model_every=1000, # model with which qvals are calculated
                       learn_epsilon_half_life=3000, discount_factor=0.9,
                       mem_size=1000, mem_batch_size=4, mem_bias_prob=0.9,
                       *args,**kwargs):
        super().__init__(*args,**kwargs)

        self.target_model = target_model
        self.update_target_model_every = update_target_model_every

        self.discount_factor = discount_factor
        self.learn_epsilon = 1 # for epsilon-greedy search
        self.learn_epsilon_half_life = learn_epsilon_half_life # time it takes to fall to half
        self.memory = MemoryBuffer(size=mem_size, batch_size=mem_batch_size, bias_prob=mem_bias_prob)
        self.prev_state = self.get_game_state()
        self.prev_score = 0

        # info vars
        self._real_event = ""
        self._reward = 0
        self._last_q = 0
        self._loss = 0

    def step(self):
        if self.learn_epsilon_half_life:
            hl_ratio = self.step_counter/self.learn_epsilon_half_life
            self.learn_epsilon = math.pow(2, -hl_ratio)
        # update target model
        if self.step_counter % self.update_target_model_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())


        self.count_step() # computes score and counts step

        # compute reward
        reward = self.compute_reward()

        qvals_prev = self.model(self.prev_state)
        qvals_prev = qvals_prev.detach()
        if self.learn_epsilon_half_life and random.random() < self.learn_epsilon: # random action
            chosen_idx = random.randrange(len(globals.EVENTS))
        else: # pick best action
            chosen_idx = np.argmax(qvals_prev.numpy())

        self.curr_state = self.get_game_state()
        self.memory.add(self.prev_state,chosen_idx,reward, self.curr_state)

        chosen_event = globals.EVENTS[chosen_idx]
        self._real_event = globals.EVENTS[np.argmax(qvals_prev.numpy())] # for info

        if chosen_event != self.last_event:
            self.vm.reset_keys()
            self.vm.keyDown(chosen_event)

        self.last_event = chosen_event
        self._reward = reward # for info

        if len(self.memory.buffer) >= self.memory.size:
            self.learning_step()

        self.prev_state = self.curr_state


    def learning_step(self):
        self.optimizer.zero_grad()
        batch = self.memory.get_batch()

        # implement pseudo-Double-DQN
        # |Q|(s',a')
        with torch.no_grad():
            tqvals_curr = self.target_model(batch["curr_state"]) # qvals for all possible actions for curr

            # Q(s',a')
            qvals_curr = self.model(batch["curr_state"])
            argmax_qval_curr = torch.argmax(qvals_curr.detach(),dim=1)

        #self._last_q = float(max_qval_curr[0])
        # Q(s,a)
        qvals_prev = self.model(batch["prev_state"]) # Shape = (b,8)


        # don't touch actions that weren't activated
        target = torch.tensor(qvals_prev).detach()

        for i,prev_a in enumerate(batch["prev_action"]):
            argmax_curr = argmax_qval_curr[i]
            target[i][prev_a] = batch["reward"][i] + self.discount_factor*tqvals_curr[i][argmax_curr]

        loss = self.loss_func(qvals_prev, target)
        self._loss = float(loss)
        loss.backward()

        if batch["reward"][0] > 0:
            self.positive_step_counter += 1

        self.optimizer.step()
