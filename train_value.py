SAVED_EVERY_EPOCHS = 10
VAL_PERC = 0.05

import os, random, time, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import globals
import model_creator
import matplotlib.pyplot as plt

#random.seed(42)

# configure tensorboard logging (pyplot too slow for real-time logging)
from tensorboard_logger import configure, log_value
timestamp = str(time.time())[-6:]
configure(globals.DIR + "/runs/value_net_"+timestamp, flush_secs=5)

train_data = []
root_dir = globals.DIR + "/value_net_train"

REGRESSOR = globals.VALUENET_REGRESSOR
if REGRESSOR:
    DIRS = [d for d in os.listdir(root_dir) if os.path.isdir(root_dir + "/" + d)]
else:
    DIRS = globals.VALUENET_DIRS

for label_class, str_label in enumerate(DIRS):
    if REGRESSOR: real_label = float(str_label)/100 # [0,1]

    for im_fname in os.listdir(root_dir + "/" + str_label):
        im_path = root_dir + "/" + str_label + "/" + im_fname
        im = Image.open(im_path).convert("L")
        arr = np.array(im.getdata()).astype(np.float32)
        arr *= (1.0/255.0) # normalize to [0,1]
        arr = torch.Tensor(arr.reshape((1, 1, im.size[1], im.size[0]))) # (1, Ch,H,W)

        if REGRESSOR:
            label = torch.Tensor( ((real_label,),) ) # 1x1 matrix
        else:
            # one-hot encoding
            label = [0]*len(DIRS)
            label[label_class] = 1
            label = torch.Tensor(label).view(-1,len(DIRS)) # 1x10 matrix

        train_data.append((arr,label))

random.shuffle(train_data)

## Put train data in batches
batch_size = 32
batched_train_data = []
for si in range(0, len(train_data), batch_size):
    try:
        tensors = []
        labels  = []
        for bi in range(batch_size):
            tensors.append(train_data[si+bi][0])
            labels.append(train_data[si+bi][1])
        new_tensor = torch.cat(tuple(tensors))
        new_labels = torch.cat(tuple(labels))
        batched_train_data.append((new_tensor,new_labels))

    except IndexError:
        pass


val_batch_num = math.floor(len(batched_train_data)*VAL_PERC)
random.shuffle(batched_train_data)
train_data = batched_train_data[val_batch_num: ]
val_data = batched_train_data[0:val_batch_num]

model = model_creator.create_score_model()

#model.load_state_dict(torch.load(globals.DIR + "/ValueNet.model"))

optimizer = optim.Adadelta(model.parameters(), lr=0.1)

criterion = nn.MSELoss() if REGRESSOR else nn.BCEWithLogitsLoss()

current_epoch = 0
while True:
    epoch_losses = []
    random.shuffle(train_data)

    for bi,batch in enumerate(train_data):
        optimizer.zero_grad()
        result = model(batch[0])
        loss = criterion(result, batch[1])
        loss.backward()
        optimizer.step()

        epoch_losses.append(float(loss)) # casting to float prevents autograd

        _s = str(round((bi/len(train_data))*100, 2))
        print("Epoch: " + str(current_epoch) + " | "  + _s + "% | ", end = "\r")

    # epoch finished
    ep_loss = sum(epoch_losses)/len(epoch_losses)
    log_value("Loss", ep_loss, current_epoch)

    # get validation loss
    val_loss = 0
    for bi,batch in enumerate(val_data):
        with torch.no_grad():
            result = model(batch[0])
            loss = criterion(result, batch[1])
            val_loss += float(loss)
    val_loss /= len(val_data)
    log_value("Validation Loss", val_loss, current_epoch)

    current_epoch += 1

    if current_epoch % SAVED_EVERY_EPOCHS == 0: # save model
        torch.save(model.state_dict(), globals.DIR + "/models/ValueNet_"+str(current_epoch))
