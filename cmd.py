import sys,os,time
import globals
import vm_intf

vm = vm_intf.VMInterface(globals.VNC_IP, display=0, password=globals.VNC_PASS)

if len(sys.argv) > 1:
    if sys.argv[1] == "-resetkeys":
        for e in globals.EVENTS:
            vm.keyUp(e)
        print("Reset all keys!")
        sys.exit()
    elif sys.argv[1] == "-screenshot":
        vm.refreshScreen()
        vm.get_game_screen(return_im=True).save(globals.DIR + "/screen.png")
        sys.exit()
