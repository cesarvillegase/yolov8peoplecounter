import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone


model=YOLO('yolov8n.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('Counter')
cv2.setMouseCallback('Counter', RGB)
cap=cv2.VideoCapture('video.mp4')


my_file = open("coco.txt", "r")
#my_file = open("labels.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
person_entering={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cx1 = 480 
cx2 = 400 
offset=6


while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        
        if cx1 < (cx + offset) and  cx1 > (cx - offset):
            cv2.rectangle(frame, (x3, y3), (x4,y4), (0,0,255),2)
            cvzone.putTextRect(frame,f'{id}', (x3, y3),1,2)
            person_entering[id] = (cx, cy)
             
        if id in person_entering:
            if cx2 < (cx + offset) and  cx2 > (cx - offset):
                cv2.rectangle(frame, (x3, y3), (x4,y4), (0,255,255),2)
                cvzone.putTextRect(frame,f'{id}', (x3, y3),1,2)
                if counter1.count(id) == 0:
                    counter1.append(id)
            
    
    #cv2.line(frame, (start_x, start_y), (end_x, end_y), (B, G, R), thickness)
    
    # Vertical Line
    cv2.line(frame,(cx1, 0),(cx1,1080),(0,255,0),2) # Green
    cv2.line(frame,(cx2, 0),(cx2,1080),(0,255,255),2) # Yellow
    
    # Horizontal line
    #cv2.line(frame,(0, cy1),(1080,cy1),(0,255,0),2) # Green
    #cv2.line(frame,(0, cy2),(1080,cy2),(0,255,255),2) # Yellow

    entering = (len(counter1))
    cvzone.putTextRect(frame, f'Person {entering}', (50, 60), 2, 2)
    cv2.imshow("Counter", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()