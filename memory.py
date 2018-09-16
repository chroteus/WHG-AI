from collections import namedtuple
import random
import numpy as np
import torch

Memory = namedtuple("Memory", ["prev_state","prev_action","reward","curr_state"]) # SARS

class MemoryBuffer:
    def __init__(self, size=1000, batch_size=4, replace=True, bias_prob=0):
        self.buffer = []
        self.size = size
        self.batch_size = batch_size
        # whether to replace into buffer after sampling buffer
        # DQAgent should replace, PAgent shouldn't
        self.replace = replace

        # prefer memories with higher reward
        # or pick random memory with p=(1-bias_prob)
        self.bias_prob = bias_prob

    def add(self, prev_state,prev_action,reward,curr_state):
        self.buffer.append(Memory(prev_state,prev_action,reward,curr_state))
        while len(self.buffer) > self.size:
            #self.buffer.sort(key=lambda b:b.reward) # remove examples with rewards last
            self.buffer.pop(0)

    def get_batch(self):
        # random.shuffle(self.buffer)
        # result = self.buffer[-self.batch_size:]
        # self.buffer = self.buffer[:-self.batch_size]
        #
        # return result

        if random.random() < self.bias_prob:
            # softmax
            p_dist = np.array([abs(mem.reward) for mem in self.buffer])

            p_dist = np.exp(p_dist)
            p_dist /= sum(p_dist)
            _idx = np.arange(len(self.buffer))
            batch_list_idx = np.random.choice(_idx, size=self.batch_size, p=p_dist)
        else:
            batch_list_idx = random.sample(range(len(self.buffer)),self.batch_size) # return indices so we can pop

        batch = {
            "prev_state":[],
            "prev_action":[],
            "reward":[],
            "curr_state":[],
        }
        for bi in batch_list_idx:
            b = self.buffer[bi]
            # b is the Memory - a namedtuple
            s,a,r,s_p = b
            batch["prev_state"].append(s)
            batch["prev_action"].append(a)
            batch["reward"].append(r)
            batch["curr_state"].append(s_p)


        if not self.replace:
            for bi in sorted(batch_list_idx, reverse=True):
                self.buffer.pop(bi)

        for k in ("prev_state", "curr_state"):
            batch[k] = torch.cat(batch[k],dim=0)

        return batch
