from cv2 import cv2
import dlib
import keyboard

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dl_face=dlib.get_frontal_face_detector()
dl_facelandmark=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

eye=cv2.CascadeClassifier('haarcascade_eye.xml')
nose=cv2.CascadeClassifier('nose.xml')
mouth=cv2.CascadeClassifier('mouth.xml')

#cap=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('out.avi',fourcc, 30.0, (1280,720))
video_capture = cv2.VideoCapture("20210115_011257.mp4")

while(video_capture.isOpened()):
    #success, img=cap.read() 
    success,img=video_capture.read()
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #используя opencv и каскадные файлы
    faces=face.detectMultiScale(img_gray, 1.1, 19)#список координат найденых лиц    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+h,y+w), (0,255,0), 2)    
        img_face=img_gray[y:y+h,x:x+h]    

        eyes=eye.detectMultiScale(img_face,1.1,19)
        noses=nose.detectMultiScale(img_face,1.1,19)
        mouths=mouth.detectMultiScale(img_face,1.1,19)

        for(ex,ey,eh,ew) in eyes:
            cv2.circle(img,(x+ex+eh//2,y+ey+ew//2),4,(0,0,255),-1)   
             
        for(nx,ny,nh,nw) in noses:
            cv2.circle(img,(x+nx+nh//2,y+ny+nw//2), 4,(0,0,255),-1) 
            
        for(mx,my,mh,mw) in mouths:
            cv2.circle(img,(x+mx+mh//2,y+my+mw//2), 4,(0,0,255),-1) 
    #используя dlib 
    faces2=dl_face(img_gray)
    for elem in faces2:
        face_landmark=dl_facelandmark(img_gray,elem)

        for i in range(0,68):
            x=face_landmark.part(i).x
            y=face_landmark.part(i).y
            cv2.circle(img, (x,y), 4, (255,0,0), -1)
    if success:          
        out.write(img)        
    #cv2.imshow('rez',img)
    if cv2.waitKey(1) and 0xff==ord('q'):
        break
    if keyboard.is_pressed('q'):
        break
#cap.release()
video_capture.release()
