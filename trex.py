import numpy as np
import cv2
import time
import mss
import pyautogui

def run_dino(array, y, x, fps):
    delta = fps + 70
    array = array[y+1, x:x+delta]
    return np.unique(array).shape[0]

def recognize_template(img, temp):
    coeff = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    max_coeff = np.max(coeff) 
    if(max_coeff < 0.7): 
        return None
    
    y, x = np.unravel_index(np.argmax(coeff), coeff.shape)
    return (y, x)

def play_the_game():
    dino_temp = cv2.imread("templates/dino.jpg", cv2.IMREAD_GRAYSCALE)
    game_over = cv2.imread("templates/game_over.jpg", cv2.IMREAD_GRAYSCALE)
    
    width, height = pyautogui.size()
    mon = (0, 0, width, height, 2)
    
    img = cv2.cvtColor(np.asarray(mss.mss().grab(mon)), cv2.COLOR_BGR2GRAY)
    
    coords = recognize_template(img, dino_temp)
    
    y = coords[0] + int(dino_temp.shape[0] / 2) + 10
    x = coords[1] + int(dino_temp.shape[1] / 2) + 30

    fps = 0
    
    while True:
        screen = cv2.cvtColor(np.asarray(mss.mss().grab(mon)), cv2.COLOR_BGR2GRAY)
        
        if run_dino(screen, y, x, fps) > 1:
            pyautogui.press('up')
            fps += 1
        if recognize_template(screen, game_over):
            fps = 0
            return "Game over!"

time.sleep(5)
print(play_the_game())