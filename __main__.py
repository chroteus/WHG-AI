import sys,os,time,glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model_creator
import vm_intf
import globals
from dqagent import DQAgent
from pagent import PAgent

# tensorboard logging
from tensorboard_logger import configure, log_value

timestamp = str(round(time.time()))[-6:]
if globals.QNET_ENABLED:
    print("===================")
    print("==| Using QNet! |==")
    print("===================")
    run_dir = globals.DIR + "/runs/q_net_"+timestamp
else:
    print("========================")
    print("==| Using PolicyNet! |==")
    print("========================")
    run_dir = globals.DIR + "/runs/p_net_"+timestamp
configure(run_dir, flush_secs=5)

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)
# focus on the game
vm.mouseMove(200,200)
vm.mousePress(1)
vm.refreshScreen()

# create q net
agent_net = model_creator.create_model()
target_qnet = model_creator.create_model() # same as qnet!!

if len(sys.argv) > 1:
    if sys.argv[1] == "-resume":
        print("Resuming...")
        fullpath = os.path.join(globals.DIR, "models")
        all_models = glob.glob(fullpath + "/*")

        try:
            latest_model = max(all_models, key=os.path.getctime)
            agent_net.load_state_dict(torch.load(latest_model))
            target_qnet.load_state_dict(torch.load(latest_model))

            _, _modelname = os.path.split(latest_model)
            print("[Loaded model: " + _modelname + "]")

        except ValueError as f: # no models
            print("!!! No models found, using a fresh one! !!!")

# load value_net
value_net = model_creator.create_score_model()
value_net.load_state_dict(torch.load(globals.DIR + "/ValueNet.model"))
value_net.train(False) # disable dropout
# init agent
if globals.QNET_ENABLED:
    agent = DQAgent(model=agent_net, target_model=target_qnet,
                    value_net=value_net, vm=vm,
                    lr=0.001,
                    update_target_model_every=10000,
                    score_epsilon=0.02,
                    avg_score_buffer_size=6, #1
                    round_score_dp=2, # 2 dp
                    learn_epsilon_half_life=False, # using NoisyLinear layers set to False
                    discount_factor=0.99,
                    mem_size=8000,
                    past_score_horizon=3, #5
                    mem_batch_size=8,
                    mem_bias_prob=0.9)
else:
    agent = PAgent(model=agent_net,value_net=value_net,vm=vm,
                   lr=0.01,
                   score_epsilon=0.02,
                   round_score_dp=2,
                   chance_coeff=0,
                   #chance_coeff_hl = 5000,
                   mem_size=5000,
                   past_score_horizon=3, #5
                   mem_batch_size=8,
                   avg_score_buffer_size=5,
            )

print("=== Loaded everything! ===")

SAVED_EVERY_STEPS = 200 # saves progress every ~5 minutes
latest_score = 0
max_score = 0
last_100_scores = []

max_score = 0
time_of_last_step = time.time()-globals.TIMESTEP
while True:


    time_before_net = time.time()
    vm.refreshScreen()
    agent.step()
    time_after_net = time.time()

    latest_score = agent.score

    # end step
    print("-------")
    print(agent.last_event)
    print("["+agent._real_event+"]")
    print("Score: " + str(round(agent.score, 5)))
    print("Reward: " + str(round(agent._reward,3)))
    if globals.QNET_ENABLED:
        pass
        #print("Q: " + str(round(agent._last_q,8)))


    last_100_scores.append(latest_score)
    if len(last_100_scores) > 100:
        max_score = max(last_100_scores)
        last_100_scores = []
        log_value("Max Score", max_score, agent.step_counter)
        log_value("Loss", agent._loss, agent.step_counter)
    #log_value("Score", latest_score, agent.step_counter)

    #log_value("Steps Pos.", agent.positive_step_counter, agent.step_counter)
    #log_value("Steps Neg.", agent.negative_step_counter, agent.step_counter)

    if agent.step_counter % SAVED_EVERY_STEPS == 0: # save model
        if globals.QNET_ENABLED:
            _dn = globals.DIR + "/models/QNet_"+str(agent.step_counter)
        else:
            _dn = globals.DIR + "/models/PNet_"+str(agent.step_counter)

        torch.save(agent_net.state_dict(), _dn)


    time_diff = min(time_after_net - time_before_net, globals.TIMESTEP)
    time_diff_str = str(round((time_after_net - time_before_net)-globals.TIMESTEP,3))
    if time_diff < globals.TIMESTEP:
        print("==| Performance: Good. (" + time_diff_str + ") |==")
    else:
        print("==!! Performance: Bad! (" + time_diff_str + ") !!==")

    time.sleep(globals.TIMESTEP-time_diff)
