
import os,sys,time
import numpy as np
from PIL import Image,ImageFilter,ImageOps,ImageChops,ImageMath
from vncdotool import api as vnc_api
import pytesseract
import globals

class VMInterface:
    def __init__(self, ip, display="0",password=None):
        self.client = vnc_api.connect(str(ip)+":"+str(display), password=password)

    def get_game_screen(self, box=(10,35,605,385), return_im=False):
        assert self.client.screen, "Call vm.refreshScreen() before trying to get image input!"
        im = self.client.screen.crop(box)
        im = im.resize((globals.IMAGE_WIDTH,globals.IMAGE_HEIGHT), resample=Image.BICUBIC)

        if return_im:
            return im
        else:
            arr = np.array(im.getdata()).astype(np.float32)
            arr *= (1.0/255.0) # normalize

            return arr.reshape((1, 3, im.size[1], im.size[0])) # (1, C,H,W)

    def get_deaths_and_level(self, box=(280,10, 590,35)):
        im = self.client.screen.crop(box).convert("L")
        s = pytesseract.image_to_string(im, config="--psm 7 -l eng")
        level = int(s.split("/")[0])
        deaths = int(s.split(":")[1])
        return (deaths,level)


    # intercept missing methods and call them from self.client
    def __getattr__(self, name):
        return self.client.__getattr__(name)
