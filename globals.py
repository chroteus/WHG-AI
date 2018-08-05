IMAGE_WIDTH,IMAGE_HEIGHT = 96,56#144,84# # specify this for model to later access
IMAGE_CHANNELS = 3
OUTPUT_NUM = 8
EVENTS = ["up","down","left","right","up-left","up-right","down-left","down-right"]
EVENTS_IDX = {x:i for i,x in enumerate(EVENTS)} # prevent calling .index on EVENTS in tight loops
TIMESTEP = 0.2

import os
DIR = os.path.dirname(os.path.realpath(__file__))

VNC_IP   = "192.168.0.103"
VNC_PASS = "root"

VALUENET_REGRESSOR = True # Whether to use classifier or regressor mode
VALUENET_DIRS = ["0","10","20","30","40","50","60","70","80","90"] # Used so that test_value can access same labels
