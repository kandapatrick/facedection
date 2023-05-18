import inline as inline
import matplotlib as matplotlib
import numpy as np
import cv2 as cv2


#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Routine to fix
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
face_cascade = cv2.CascadeClassifier()
eye_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile('./model_data/haarcascade_frontalface_alt.xml')):
    print('--(!)Error loading face cascade')
if not eye_cascade.load(cv2.samples.findFile('./model_data/haarcascade_eye.xml')):
    print('--(!)Error loading eye cascade')
image = cv2.imread('three.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Equalising Histograms increases the contrasts which improves
gray = cv2.equalizeHist(gray)
plt.imshow(fixColor(gray))

scaleFactor = 1.3
minWindows = 5
faces = face_cascade.detectMultiScale(gray, scaleFactor, minWindows)

for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = gray[y:y+h,x:x+w]
    #Detect eyes in the face
    eyes = eye_cascade.detectMultiScale(faceROI)
    for (x2,y2,w2,h2) in eyes:
        eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        image = cv2.circle(image, eye_center, radius, (255, 0, 0 ), 4)
plt.imshow(fixColor(image))

from imutils.video import FPS
fps=FPS()
start=fps.start()
frameCnt=0

writer = cv2.VideoWriter("output.mp4",
                         cv2.VideoWriter_fourcc(*"MP4V"), 30,(640,480))

#Create a new video stream and get total frame count
video_stream = cv2.VideoCapture('images/faces.mp4')
total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
total_frames

#fps = FPS().start
frameCnt=0
while(frameCnt < 300):
    #print (frameCnt)
    frameCnt+=1
    ret, frame = video_stream.read()

    # Convert current frame to grayscale
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gframe = cv2.equalizeHist(gframe)
    eyes = eye_cascade.detectMultiScale(gframe, 1.3, 5)
    if len(eyes) > 0:
        for (x,y,w,h) in eyes:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    writer.write(cv2.resize(frame, (640,480)))
    fps.update() #update fps

fps.stop()

print("FPS: {:.3f}".format(fps.fps()))

#Release video object
video_stream.release()
writer.release()



