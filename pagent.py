import random, math
import numpy as np
import globals
import torch
from agent import Agent
from memory import MemoryBuffer

LEN_EVENTS = len(globals.EVENTS)

class PAgent(Agent):
    def __init__(self, chance_coeff=1, chance_coeff_hl=False,
                 mem_size=1000, mem_batch_size=4,
                 penalize_if_repeats=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_func = torch.nn.KLDivLoss()
        # Whether to just choose max prob. event (0.0) or pick from probability
        # distribution produced by PolicyNet (1.0).
        # It's a float, so you can control how much exploring should be done by agent
        # chance_coeff_hl: If defined, makes it a decaying variable, with halflife in steps
        self.chance_coeff = chance_coeff
        self.chance_coeff_hl = chance_coeff_hl

        # memory buffer
        self.memory = MemoryBuffer(size=mem_size, batch_size=mem_batch_size, replace=False)

        self.curr_state = self.get_game_state()
        self.prev_state = self.curr_state

        self.max_event = ""
        self._reward = 0
        self._loss = 0

    def step(self):
        if self.chance_coeff_hl:
            hl_ratio = self.step_counter/self.chance_coeff_hl
            self.chance_coeff = math.pow(2, -hl_ratio)

        self.count_step()
        reward = self.compute_reward()
        self.curr_state = self.get_game_state()

        probs = self.model(self.curr_state)
        # p is the probability distribution set by policy net
        probs_list = probs.detach().numpy()[0]
        self.max_idx = np.argmax(probs_list)
        self._real_event = globals.EVENTS[self.max_idx]

        if random.random() > self.chance_coeff:
            chosen_idx = self.max_idx
        else:
            chosen_idx = np.random.choice(range(LEN_EVENTS), p=probs_list)

        chosen_event = globals.EVENTS[chosen_idx]
        if chosen_event != self.last_event:
            self.vm.reset_keys()
            self.vm.keyDown(chosen_event)

        self.memory.add(self.prev_state,chosen_idx,reward, self.curr_state)

        self.last_event = chosen_event
        self._reward = reward # for info
        if len(self.memory.buffer) >= self.memory.size:
            self.learning_step()

        self.prev_state = self.curr_state

    def learning_step(self):
        self.optimizer.zero_grad()
        batch = self.memory.get_batch()
        probs = self.model(batch["prev_state"]) # (b,8)
        target = torch.tensor(probs).detach() # copy without grad

        for i,p in enumerate(probs):
            prev_a = batch["prev_action"][i]
            if batch["reward"][i] > 0:
                target[i] = torch.zeros(LEN_EVENTS)
                target[i][prev_a] = 1.0
            else:
                target[i][prev_a] = 0.0

        loss = self.loss_func(probs, target)
        self._loss = float(loss)
        loss.backward()
        self.optimizer.step()

    def __learning_step(self):
        if len(self.event_buffer) > 0:
            self.optimizer.zero_grad()

            target = torch.zeros((len(self.event_buffer),LEN_EVENTS))
            probs_list = []
            for i,event_probs in enumerate(self.event_buffer):
                event,probs = event_probs
                event_idx = globals.EVENTS_IDX[event]
                target[i][event_idx] = 1
                probs_list.append(probs)

            probs_batch = torch.cat(probs_list)
            loss = self.loss_func(probs_batch, target)
            loss.backward()
            self.optimizer.step()
            # end learning step
            self.clear_event_past()
            self.end_learning_step()

            self.positive_step_counter += 1
            print("POSITIVE STEP +++")

    def negative_step_for_repeating(self): # used for penalizing if agent repeats itself
        self.optimizer.zero_grad()
        for event,probs in self.event_buffer:
            event_idx = globals.EVENTS_IDX[event]
            # probability distribution where event we're penalizing is 0
            target = torch.tensor(probs) # copy as tensor
            target += target[0][event_idx]/(LEN_EVENTS-1) # make sure all elements add up to 1 after we set unwanted event to 0
            target[0][event_idx] = 0

            loss = self.loss_func(probs, target)

            loss.backward()

            self.negative_step_counter += 1
            print("NEGATIVE STEP ---")
            self.clear_event_past()
        self.optimizer.step()

    def clear_event_past(self):
        self.event_buffer = []
        self.event_history = []
