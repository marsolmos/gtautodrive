import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, W, A, S, D


def roi(img, vertices):
    mask = np.zeros_like(img) # blank mask
    cv2.fillPoly(mask, vertices, 255) # fill the mask
    masked = cv2.bitwise_and(img, mask) # now only show the area in the mask
    return masked


def draw_lines(img,lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[0,250],[300,150],[400,150],[800,340],[800,450],
                        [0,450]], np.int32)

    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    processed_img = roi(processed_img, [vertices])
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 550, 10)
    draw_lines(processed_img,lines)
    return processed_img


def main():
    while(True):
        # PressKey(W)
        # 800x600 windowed mode
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        # cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
