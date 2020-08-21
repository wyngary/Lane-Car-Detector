import matplotlib.pylab as plt
from cv2 import cv2
import numpy as np


# Function that find the region of lane 
def region_of_interest(img,vertices):
    mask=np.zeros_like(img)
    match_mask_color=255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def draw_the_line (img,lines):
    img=np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=10)
    
    img=cv2.addWeighted(img, 0.8, blank_image,1,0.0)
    return img

#image=cv2.imread('111.jpg')
#image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
# Define region of interest
    height=image.shape[0]
    width=image.shape[1]
    region_of_interest_vertirces=[
        (0,height),
        (width/4,2*height/3),
        (3*width/4,2*height/3),
        (width,height)
    ]

    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    canny_image=cv2.Canny(gray_image,100,200)
    cropped_iamge=region_of_interest(canny_image,np.array([region_of_interest_vertirces],np.int32),)
    lines=cv2.HoughLinesP(
        cropped_iamge,
        rho=6,
        theta=np.pi/180,
        threshold=30,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=50
    )

    image_with_lines=draw_the_line(image,lines)
    return image_with_lines

cap=cv2.VideoCapture('Lane Video.mp4')

ret, frame1=cap.read()
frame1=process(frame1)
ret, frame2=cap.read()
frame2=process(frame2)

def car (frame1,frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    #cv2.drawContours(frame1,contours, -1, (0,255,0),2)
    return frame1

while cap.isOpened() :
    frame1=car(frame1,frame2)
    cv2.imshow('frame',frame1)
    frame1=frame2
    ret, frame2=cap.read()
    frame2=process(frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#plt.imshow(image_with_lines)
#plt.show()