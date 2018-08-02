import sys,os,time
import vm_intf
import globals

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)

step = 0

while True:
    time.sleep(0.7)
    step += 1
    vm.refreshScreen()
    vm.get_game_screen(return_im=True).save(globals.DIR + "/raw_train_data/" + str(time.time())[-7:] + ".png")
    print(str(step), end="\r")
