import cv2
vidcap = cv2.VideoCapture('./video/19.Xiaomi FIMI X8 SE 2020 DRONE vs Mavic AIR DRONE FLIGHT TEST.mp4') ## 다운받은 비디오 이름 
success,image = vidcap.read()
count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    
    if count == 10000: # 종료 시점 
        break

    if(int(vidcap.get(1)) % 40 == 0): # 20 프레임당 저장 
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("frame%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1