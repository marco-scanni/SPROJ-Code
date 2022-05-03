#!/usr/bin/env python
# coding: utf-8

# In[154]:


# 0.a install Dependencies
#----------------------------#
get_ipython().system('pip install mediapipe opencv-python')


# In[1]:


#0.b import Dependencies
#-----------------------------#
import cv2
import mediapipe as mp
import numpy as np #use for trig when calculating angles between kp's
import time
mp_drawing = mp.solutions.drawing_utils #to draw poses
mp_pose = mp.solutions.pose #imports 'pose' model from mediapipe


# In[15]:


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #capture device

font = cv2.FONT_HERSHEY_DUPLEX
red    = (0,0,255)
green  = (0,255,0)
blue   = (243,247,0)
orange = (24,197,245)

timer = int(3)

#scapula positioning variables
color_scap, stage_scap, max_depression = blue, None, 0

#hinge variables
color_hinge, stage_hinge, eye_ear_level_angle = blue, None, 0

#ASI variables
color_asi, stage_asi, chin_down_nose_height = blue, None, 0
    
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
        
        #mediapipe requires RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False #saves memory
        
        #give results of detection
        results = pose.process(image)
        
        #Go back to BGR for cv
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Get KPs
        try:
            keypoints = results.pose_landmarks.landmark
            
            # Get coordinates for display
            r_shoulder = get_coordinates_of('r_shoulder') #relevant kp's for scapular height
            
            r_eye = get_coordinates_of('r_eye')           #relevant kp's for neck hinge
            r_ear = get_coordinates_of('r_ear')
            
            nose = get_coordinates_of('nose')             ##relevant kp's for atlantostyloid interval
            
            
            #RealTime Scapula Height calculation and display
            rshoulder_height = calculate_height_of(r_shoulder)

            cv2.putText(image, str(rshoulder_height), 
                np.subtract(tuple(np.multiply(r_shoulder, [640, 480]).astype(int)),(10,10)), 
                    font, 0.6,(color_scap), 1, cv2.LINE_AA
                )
    

            
            #RealTime Horizon Level calculation and display
            r_eye_ear_angle = np.rint(calculate_angle(r_eye,r_ear,[r_ear[0],0]))

            cv2.putText(image, str(r_eye_ear_angle), 
                           np.subtract(tuple(np.multiply(r_ear, [640, 480]).astype(int)),(10,10)), 
                           font, 0.6,(color_hinge), 1, cv2.LINE_AA
                       )
            
            #RealTime Nose Height calculation and display
            nose_height = calculate_height_of(nose)

            cv2.putText(image, str(nose_height), 
                    np.add(tuple(np.multiply(nose, [640, 480]).astype(int)),(10,10)), 
                    font, 0.5,(color_asi), 1, cv2.LINE_AA
                        )
            
            #Header for Ideal Threshold Values:
            if max_depression != 0 or eye_ear_level_angle != 0 or chin_down_nose_height != 0:
                
                cv2.putText(image, "Ideal Ranges", 
                        (490,50), 
                        font, 0.65,green, 1, cv2.LINE_AA
                            )


#-------------------SCAPULA----------------------------------#

            capkey_scap = cv2.waitKey(10)
            if capkey_scap == ord('s'):
                prev = time.time()


                while timer >=0:
                    ret,image = cap.read()
                    
                    cv2.putText(image, str(timer),
                                (300,240), 
                                font,7, (255,0, 0), 4, cv2.LINE_AA)
                    cv2.imshow('SPROJ Cam Feed', image)
                    
                    cv2.putText(image, "Put Scapula in Maximal Depression",
                                (30,50), 
                                font,1, (255,0, 0), 4, cv2.LINE_AA)
                    
                    cv2.imshow('SPROJ Cam Feed', image)                    
                    cv2.waitKey(1)
                        
                    cur = time.time()
                        
                    if cur-prev >=1:
                        prev = cur
                        timer = timer-1
                
                else:
                    ret,image = cap.read()
                    cv2.imshow('SPROJ Cam Feed', image)
                    
                    results = pose.process(image)
                    keypoints = results.pose_landmarks.landmark
                    r_shoulder = get_coordinates_of('r_shoulder')
                    
                    cv2.waitKey(1000) 
                    timer = int(3)
                    max_depression = calculate_height_of(r_shoulder)
            
            

            if max_depression != 0:
                
                #Header
                cv2.putText(image,'Scapula:', 
                              (330,80), 
                               font, 0.55,color_scap, 1, cv2.LINE_AA
                           )
                #Ideal Range
                cv2.putText(image,"< "+ str(max_depression - 5), 
                              (500,80), 
                               font, 0.55,color_scap, 1, cv2.LINE_AA
                           )
            
                #scapular threshold ranges
                if rshoulder_height > max_depression  - 5:
                    stage_scap = "Too Low"
                    color_scap = red
                else:
                    stage_scap = "Adequate Height"
                    color_scap = green
            else:
                stage_scap = "Not Yet Calibrated"
                color_scap = blue
                
                
            #Scapular threshold displays
            
            #Header
            cv2.putText(image, "Scapular Height Status:", 
                          (10,370), 
                           font, 0.7,(color_scap), 1, cv2.LINE_AA
                       )
           #Status
            cv2.putText(image,stage_scap, 
                          (330,370), 
                           font, 0.7,(color_scap), 1, cv2.LINE_AA
                       )
            
            
#-----------------END--SCAPULA--------------------------------------#
#-----------------CERVICAL HINGE------------------------------------#

    
            capkey = cv2.waitKey(10)
            if capkey == ord('n'):
                prev = time.time()
                
                while timer >=0:
                    ret,image = cap.read()
                    
                    cv2.putText(image, "Lift Back of Head Towards Ceiling",
                                (30,50), 
                                font,1, (255,0, 0), 4, cv2.LINE_AA)
                    
                    cv2.putText(image, str(timer),
                                (320,240), 
                                font,7, (255,0, 0), 4, cv2.LINE_AA)
                    cv2.imshow('SPROJ Cam Feed', image)
                    cv2.waitKey(1)
                        
                    cur = time.time()
                        
                    if cur-prev >=1:
                        prev = cur
                        timer = timer-1
                
                else:
                    ret,image = cap.read()
                    cv2.imshow('SPROJ Cam Feed', image)
                    
                    results = pose.process(image)
                    keypoints = results.pose_landmarks.landmark
                    r_eye = get_coordinates_of('r_eye')
                    r_ear = get_coordinates_of('r_ear')
                    nose  = get_coordinates_of('nose')
                    
                    cv2.waitKey(1000) 
                    timer = int(3)
                    eye_ear_level_angle = np.rint(calculate_angle(r_eye,r_ear,[r_ear[0],0]))
                    chin_down_nose_height = calculate_height_of(nose)
        
            
                    
                    
            #hinge threshold ranges

            if eye_ear_level_angle != 0:
                
                #Header
                cv2.putText(image,'Eye-Ear Level:', 
                              (330,110), 
                               font, 0.55,color_hinge, 1, cv2.LINE_AA
                           )
                #Ideal Range
                cv2.putText(image,str(eye_ear_level_angle - 12) + ' - ' + str(eye_ear_level_angle + 8), 
                              (500,110), 
                               font, 0.55,color_hinge, 1, cv2.LINE_AA
                           )
                
                if  eye_ear_level_angle > 110 or eye_ear_level_angle < 70:
                    stage_hinge = "Bad Calibration, Try Again"
                    color_hinge = blue
            
                elif  r_eye_ear_angle > eye_ear_level_angle + 8:
                    stage_hinge = "Overly Tucking Chin"
                    color_hinge = red
                #hinging is granted 4 degrees of freedom because a slight change is inevitable when 
                #raising chin to increase ASI
                elif r_eye_ear_angle < eye_ear_level_angle - 12:
                    stage_hinge = "Hinging"
                    color_hinge = red
            
                elif stage_asi == "May Be Slightly Hinging" and stage_hinge != "Hinging":
                    stage_hinge = stage_asi
                    color_hinge = orange
                                                  
                elif (r_eye_ear_angle <= eye_ear_level_angle + 8) and (r_eye_ear_angle >= eye_ear_level_angle - 12):
                    stage_hinge = "Adequate Position"
                    color_hinge = green
            
            else:
                stage_hinge = "Not Yet Calibrated"
                color_hinge = blue
                

            
                # Hinge threshold display
            cv2.putText(image,"Cervical Hinge Status:", 
                          (10,400), 
                           font, 0.7,(color_hinge), 1, cv2.LINE_AA
                       )

            cv2.putText(image,stage_hinge, 
                          (330,400), 
                           font, 0.7,(color_hinge), 1, cv2.LINE_AA
                       )
            
         
            

#-----------------END CERVICAL HINGE---------------#

#--------------- ATLANTOSTYLOID INTERVAL------------#
     
#ASI is measured through the height of the nose keypoint AFTER properly eliminating hinge;they are closely related
#Therefore, ASI does not need it's own calibration.




            #ASI threshold ranges
            if chin_down_nose_height != 0:
            
                #Header
                cv2.putText(image,'Nose Height:', 
                              (330,140), 
                               font, 0.55,(color_asi), 1, cv2.LINE_AA
                           )
                #Ideal Range
                cv2.putText(image,str(chin_down_nose_height - 12) + " - "+ str(chin_down_nose_height - 5), 
                              (500,140), 
                               font, 0.55,(color_asi), 1, cv2.LINE_AA
                           )
            
                
                if stage_hinge == "Bad Calibration, Try Again":
                    chin_down_nose_height = 0
                
                
                elif stage_hinge == "Hinging":
                    stage_asi = stage_hinge
                    color_asi = red
                    
                elif stage_hinge == "Overly Tucking Chin":
                    stage_asi = stage_hinge
                    color_asi = red
                
                    
                elif nose_height >= chin_down_nose_height and stage_hinge != "Overly Tucking Chin":
                    stage_asi = "Gently Raise Chin"
                    color_asi = orange
                    
                #10 pixels of freedom for adequate interval or until hinge occurs
                elif nose_height < chin_down_nose_height - 15 and stage_hinge != "Hinging":
                    stage_asi = "May Be Slightly Hinging"
                    color_asi = orange
                
                elif (nose_height <= chin_down_nose_height - 5) and (nose_height >= chin_down_nose_height - 15):
                    stage_asi = "Adequate Interval"
                    color_asi = green
                
            else:
                stage_asi = "Not Yet Calibrated"
                color_asi = blue
                
                
            #ASI threshold display
            cv2.putText(image, "ASI Status:", 
                          (10,430), 
                           font, 0.7,(color_asi), 1, cv2.LINE_AA
                       )   

            cv2.putText(image, stage_asi, 
                          (330,430), 
                           font, 0.7,(color_asi), 1, cv2.LINE_AA
                       )
        
        except:
            pass
            
        
        #Render detections - pass image, kp's, and limbs between kp's
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('SPROJ Cam Feed', image)
    
        #Exit Program
        exitkey = cv2.waitKey(10) & 0xFF
        if exitkey == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[3]:


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle


# In[4]:


def calculate_height_of(a):
    ycoord = a[1]
    height = np.rint(ycoord*480)
    return height


# In[5]:


keypoint_IDs = {
    'nose':0,
    'r_eye':3,
    'r_ear':8,
    'r_shoulder':12
}


# In[6]:


def get_coordinates_of(keypoint: str):
    return[keypoints[keypoint_IDs[keypoint]].x,keypoints[keypoint_IDs[keypoint]].y]


# In[155]:





# In[ ]:




