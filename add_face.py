import face_recognition
import pickle
import cv2
import os


# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

data = {"encodings": knownEncodings, "names": knownNames}


def addface(name,data):

    cap=cv2.VideoCapture(0)
    
    i,k=0,0
    
    while True:
    
        ret,image = cap.read()   
        image=cv2.flip(image,1)
            
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,	model="hog")
            
            
        for (top, right, bottom, left) in boxes:
            	# draw the predicted face name on the image
            	cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow('Single-Threaded Detection', image)
        
        if cv2.waitKey(1) & (i>=25 or 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
         
                
        k=k+1
        if(k<4):
            continue
        
        i=i+1  
        k=0
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb)
        
         
        # loop over the encodings
        for encoding in encodings:
        		# add each encoding + name to our set of known names and
        		# encodings
        		data["encodings"].append(encoding)
        		data["names"].append(name)  
        
        
    cap.release()
    return

addface("karam",data)        

f = open("facial_data.p", "wb")
f.write(pickle.dumps(data))
f.close()        
