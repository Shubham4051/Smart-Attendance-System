#for computer-vision operations
import cv2
#for numerical calculation with accordance to cv2
import numpy as np 
#for face detection and recognition
import face_recognition as face_rec
#for system environvement traversal
import os 
#for text to speech output
import pyttsx3 as textSpeach
#for date and time sync
from datetime import datetime

engine = textSpeach.init()

# for resizing the img if needed 

'''def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

'''

path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        #img = resize(img, 0.50) (Here not required in my pc)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

#for recording the attendance records 
def MarkAttendence(name) :
    with open('attendence.csv', 'r+') as f :
        myDataList = f.readlines()
        nameList = []
        for line in myDataList :
            entry = line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList :
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            #marking of record
            f.writelines(f'\n{name}, {timestr}')
            #speech out 
            statement = str('welcome to class' + name)
            engine.say(statement)
            engine.runAndWait()

EncodeList = findEncoding(studentImg)

#vision operation via program
vid = cv2.VideoCapture(0)
while True :
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrames = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrames, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendence(name)

    #set frame name and waiting time
    cv2.imshow('Smart Attendance', frame)
    cv2.waitKey(1)