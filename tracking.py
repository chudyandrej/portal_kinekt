import numpy as np
import cv2
import time
import thread
import math
from tracked_object import TrackedObject
from collections import namedtuple
from bg_subtractor import filter_frame
from comunication import send_transaction, get_tag_permission
from antena_read import AntennaReader

###############SETTINGS##############################
URL = '/Users/andrejchudy/Desktop/video.avi'
GUI = False
PERM_RECORD = False   #permissions to record
MIN_CONTOUR_AREA = 4400
HEIGHT_TOLERANCE = 10
MAX_DISTANCE_TO_PARSE =  200
MIN_DISTANCE_TO_PARSE =  200
MAX_DISTANCE_TO_MERGE = 10 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECORD = False
dir_reversed = False    #revers monitoring
######################################################

pass_in = 0
pass_out = 0

def merge_contours(contours):
    contours = filter(lambda cnt : cv2.contourArea(cnt) > 0, contours)
    merged_contours = []
    cnts_with_centroids = []

    for cnt in contours:
        M = cv2.moments(cnt)
        # Compute centroids from the moments
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        cnts_with_centroids.append(((centroid_x, centroid_y), cnt))
    while cnts_with_centroids:
        new_blob = []
        root = cnts_with_centroids[0]
        new_blob.append(root)
        cnts_with_centroids.remove(root)
        cnts_with_centroids = merge_contour_rec(root,
                cnts_with_centroids, new_blob)
        unified_contours = [cnt[1] for cnt in new_blob]
        new_cnt = np.vstack(i for i in unified_contours)
        hull = cv2.convexHull(new_cnt)
        merged_contours.append(hull)
    return merged_contours


def merge_contour_rec(root_cnt, other_cnts, new_blob):
    near_cnts = filter(lambda cnt: is_near(root_cnt, cnt), other_cnts)
    other_cnts = filter(lambda cnt: not is_near(root_cnt, cnt), other_cnts)
    new_blob += near_cnts

    for cnt in near_cnts:
        other_cnts = merge_contour_rec(cnt, other_cnts, new_blob)
    return other_cnts

def is_near(cnt1, cnt2):
    return compute_distance(cnt1[0], cnt2[0]) < MAX_DISTANCE_TO_MERGE

def find_contours(frame, filtered_fg):
    # Function find centroids of all valid contours in the frame
    # Know issues :
    #   1.  nearby contours are not merged - considering nearness clustering

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Find contour in a given frame
    _, contours, _ = cv2.findContours(filtered_fg.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    m_conture = merge_contours(contours)
    cv2.drawContours(frame,m_conture,-1,255,-1)
    # Filter the conturs to track only the valid ones
    valid_contours = filter(is_valid_contour, m_conture)
    centroids = []
    # Iterate over valid contours
    for cnt in valid_contours:           
        # Create an empty mask
        mask = np.zeros(filtered_fg.shape,np.uint8)
        # Draw a contour on the mask
        cv2.drawContours(mask,[cnt],0,255,-1)
        #Find the minimal value of the contour (highest point)
        minVal, _, _, _ =  cv2.minMaxLoc(gray, mask=mask)
        # Threshold whole image with the range (minVal, minVal + tolerance)
        _, thresh = cv2.threshold(gray, minVal+HEIGHT_TOLERANCE,
                255, cv2.THRESH_BINARY_INV)
        # And the threshold with mask to eliminate all results not in the
        # same contour
        result = cv2.bitwise_and(mask, thresh)
        # Find biggest contour on a result
        _, contours, _ = cv2.findContours(result,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv2.contourArea)
        # Check if the biggest contour has no-zero area
        if (cv2.contourArea(max_cnt) > 0):
            #create nametuple list
            Contour_tuple = namedtuple('Contour', 'point cnt obj_count')
            # Find moments of the found contour
            M = cv2.moments(max_cnt)
            # Save centroid to point
            Point_tuple = namedtuple('Point', 'x y')
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            center_point = Point_tuple(centroid_x, centroid_y)
            # Append the tuple of coordinates to the result vector
            centroids.append(Contour_tuple(center_point, cnt,[0]))
    return centroids

def is_valid_contour(contour):
    # Function decides if the contour is good enough to be tracked
    contour_area = cv2.contourArea(contour)
    return contour_area > MIN_CONTOUR_AREA

def compute_distance(point1, point2):
    a = (point1[0] - point2[0]) 
    b = (point1[1] - point2[1]) 
    return  math.sqrt(a**2 + b**2)


def assign_centroids(tracked_objects, contours, t):
    
    distances = []  #create list of distances betwen tracked_objects and contours
    potential_relicts = []  #create list of potential_relicts contours

    for obj in tracked_objects:
        for cnt in contours:
            distance = compute_distance(obj.get_prediction(t),cnt.point)
            if(distance < MAX_DISTANCE_TO_PARSE):
                distances.append((obj, cnt, distance))
            if distance < MIN_DISTANCE_TO_PARSE:
                potential_relicts.append(cnt)

    distances = sorted(distances, key=lambda d : d[2])  #Sort from smallest
   
    seen = []

    for distance in distances:
        obj, cnt, _ = distance
        if obj not in seen:
            seen.append(obj)
            cnt.obj_count[0] += 1   #counter of objects of interest

    #distances = map(lambda d :(d[0], d[1], d[2] + d[1].obj_count[0]*penalt), distances) #penalt calculation
    #distances = sorted(distances, key=lambda d : d[2])  #sort one more

    used_objects = []
    used_cnts = []
    pairs = []

    for distance in distances:
        obj, cnt, _ = distance
        if obj not in used_objects: 
            used_objects.append(obj)    #push to used_objects
            if cnt in used_cnts:        #if more objects use the same contour
                pairs = filter(lambda p: p[1] != cnt, pairs)    #delete this conture
                used_objects = filter(lambda o: o != obj,used_objects)    #delete this conture
            else :
                pairs.append((obj, cnt))    #push to pairs

            used_cnts.append(cnt)   #push to used_cnts

    unused_objects = [obj for obj in tracked_objects if obj not in used_objects]    #select unused_objects
    unused_cnts_without_relicts = [cnt for cnt in contours 
            if cnt not in used_cnts and cnt not in potential_relicts]   #select only valid objects

    return pairs, unused_cnts_without_relicts, unused_objects

def create_objects(unused_cnts, tracked_objects, t):
    #create bjects from unused_cnts
    for cnt in unused_cnts:
        new_obj = TrackedObject(cnt.point.x, cnt.point.y, t)
        tracked_objects.append(new_obj)

def update_pairs(pairs, t):
    #update position old objects
    for pair in pairs:
        obj, cnt = pair
        obj.update(cnt.point.x, cnt.point.y, t)

def update_missing(unused_objects, tracked_objects):
    #update information about missing object
    for unused_object in unused_objects:
        if unused_object.missing() == -1:
            tracked_objects.remove(unused_object)

def counter_person_flow(tracked_objects,antenna_reader ,t):
    global pass_in
    global pass_out
    for tracked_object in tracked_objects:
        global RECORD
        RECORD = True
        if (tracked_object.start_y < FRAME_HEIGHT / 2 and 
                tracked_object.get_prediction(t).y > FRAME_HEIGHT - FRAME_HEIGHT / 4 ): #up line 
            o , i = tracked_object.abs_disto_obj(tracked_object, t)
            pass_in+=o
            pass_out+=i
            if i != 0 or o != 0:   #object-counting 
                tracked_object.start_y = FRAME_HEIGHT
                tracked_object.changed_starting_pos = True
                tag, certainity = antenna_reader.get_object_tag_id(tracked_object.center_time)
                alarm = get_tag_permission(tag)
                if dir_reversed:
                    thread.start_new_thread(send_transaction,(tag,'in',tracked_object.center_time, certainity, alarm))
                else:
                    thread.start_new_thread(send_transaction,(tag,'out',tracked_object.center_time,certainity, alarm))
            
        if (tracked_object.start_y > FRAME_HEIGHT / 2 and
                tracked_object.get_prediction(t).y < FRAME_HEIGHT / 4 ):    #down line
            o , i = tracked_object.abs_disto_obj(tracked_object, t)
            pass_in+=o
            pass_out+=i
            if i != 0 or o != 0:   #object-counting
                tracked_object.start_y = 0
                tracked_object.changed_starting_pos = True
                tag, certainity = antenna_reader.get_object_tag_id(tracked_object.center_time)
                alarm = get_tag_permission(tag)
                if dir_reversed:
                    thread.start_new_thread(send_transaction,(tag,'out',tracked_object.center_time,certainity, alarm))
                else: 
                    thread.start_new_thread(send_transaction,(tag,'in',tracked_object.center_time,certainity, alarm))

def parse_arguments(arguments):
    if "-g" in arguments:
        global GUI
        GUI = True
    if "-r" in arguments:
        global PERM_RECORD
        PERM_RECORD = True

def tracking_start(arguments):
    parse_arguments(arguments)
    # Initialise videl capture
    cap = cv2.VideoCapture(URL)
    # Take first frame as bacground reference
    _, initial_frame = cap.read()
    bg_reference = cv2.cvtColor(initial_frame,cv2.COLOR_BGR2GRAY)

    # Iterate forever
    tracked_objects = []
    if PERM_RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        record_cap = cv2.VideoWriter('output.avi',fourcc, 20.0, (FRAME_WIDTH,FRAME_HEIGHT))
    antena_reader = AntennaReader()

    while(cap.isOpened()):
        t = time.time() 
        global RECORD
        RECORD = False
        #Read frame
        ret, frame = cap.read()
        if not ret:
            print "Read failed"
            break
        # Obtain thresholded and filtered version
        filtered_fg = filter_frame(frame, bg_reference)
        # Find centroids of all contours
        centroids = find_contours(frame, filtered_fg)
        pairs, unused_centroids, unused_objects = assign_centroids(
                tracked_objects, centroids, t)
        # Create objects for centroid that were not assigned
        create_objects(unused_centroids, tracked_objects, t)
        # Update assigned centroid with current measurements
        update_pairs(pairs, t)
        # Delete missing objects and call callbacks
        update_missing(unused_objects, tracked_objects)
        # Control position all objects
        counter_person_flow(tracked_objects, antena_reader, t)
        if GUI or PERM_RECORD: 
            cv2.namedWindow('frame', 0)             #init windows
            cv2.namedWindow('filtered_fgmask', 0) 
            # Show counters
            for obj in tracked_objects:
                frame = cv2.circle(frame, obj.get_prediction(t), 5, obj.color, -1)
            
            for pair in pairs:
                obj, cnt = pair
                x, y, w, h = cv2.boundingRect(cnt.cnt)
             
                frame = cv2.rectangle(frame, (x,y), (x+w,y+h), obj.color,5)
                frame = cv2.circle(frame, obj.get_prediction(t), 10, obj.color, -1)
                frame = cv2.circle(frame, obj.get_prediction(t), MAX_DISTANCE_TO_PARSE, obj.color, 0)          
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame,str(pass_in),(10,50), font, 1,(0,0,255),2)
            frame = cv2.putText(frame,str(pass_out),(10,400), font, 1,(0,0,255),2)
        if PERM_RECORD:
            if record == True:
                record_cap.write(frame)
       
        if GUI:   
            cv2.imshow('frame',frame)
            cv2.imshow('filtered_fgmask',filtered_fg)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('s'):
                frame_delay = 500
            if key & 0xFF == ord('f'):
                frame_delay = 1

    # Teardown
    cap.release()
    if GUI:
        cv2.destroyAllWindows()
