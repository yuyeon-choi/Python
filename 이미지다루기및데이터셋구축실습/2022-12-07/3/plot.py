import cv2
import numpy as np
import matplotlib.pyplot as plt

img_ori = cv2.imread('./car1.png') 
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

def imshow(src, windowName='show', close=True):
    cv2.imshow(windowName, src)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()

def plot_images(image_list:list, title_list:list):
    img_cnt = len(image_list)
    plt.figure(figsize=(15, 15))
    for i in range(1, img_cnt+1):
        plt.subplot(1, img_cnt+1, i)
        try:
            h, w, c = image_list[i-1].shape
            if c == 3:
                plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
            else:
                continue
        except:
            plt.imshow(img_gray, 'gray')
        finally:
            plt.title(title_list[i-1])
    plt.show()