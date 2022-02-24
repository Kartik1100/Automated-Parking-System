import cv2

f=open("detect.txt","w")

cascade_source = 'cascade.xml'
video_source = 'vid.mp4'

cap = cv2.VideoCapture(video_source)
car_cascade = cv2.CascadeClassifier(cascade_source)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(True)&0xff == ord('s'):
        break

f.write("True")
 
f.close()
cv2.destroyAllWindows()
