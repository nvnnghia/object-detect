import numpy as np
from evdev import UInput, ecodes as e
import cv2
ui = UInput()
	#Check the position of the target and press the keys
        #KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_SPACE
        #KEY_W, KEY_S, KEY_D, KEY_A
i=0
cv2.waitKey(10000000)
while i<100:
    if i%10 ==0:
	ui.write(e.EV_KEY, e.KEY_A, 1)
	print("KEY_DOWN")
  	ui.write(e.EV_KEY, e.KEY_A, 0)
	ui.syn()
    if i%20 ==0:
	ui.write(e.EV_KEY, e.KEY_B, 1)
	print("KEY_DOWN")
  	ui.write(e.EV_KEY, e.KEY_B, 0)
	ui.syn()
	
    i+=1
    cv2.waitKey(1000)
    
ui.close()
