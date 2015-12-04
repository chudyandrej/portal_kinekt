import cv2
from comunication import websockets
import thread
from comunication import get_json_settings, get_list_tag

###############SETTINGS##############################
MIN_HEIGHT = 50
#####################################################

def filter_frame(frame, bg_reference):
    # Function computes threshold of the frame and filters out all noise
    # Create grayscale version of the frame and blur it
    img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Substract the bg_reference to find foreground
    fg = cv2.absdiff(img_grayscale, bg_reference)
    #Threshold the foreground
    ret, thresh = cv2.threshold(fg, MIN_HEIGHT, 255, cv2.THRESH_BINARY)
    return thresh

def load_settings():
	#load settings from server
	settings = get_json_settings()
	
 	global MIN_HEIGHT
   	MIN_HEIGHT = settings['kin_fg_minHeight']
   	
   	min_area = settings['kin_minArea']
   	max_dist_to_pars = settings['kin_maxDist_p']
   	min_dis_to_create = settings['kin_minDist_c']
   	hTolerance = settings['kin_fg_hTolerance']
   	maxDist_marge = settings['kin_fg_maxDist_marge']
   	return min_area, max_dist_to_pars, min_dis_to_create, hTolerance, maxDist_marge

def start_threads():
	#function to start all thread of program				
	get_list_tag()
	thread.start_new_thread(websockets,())	#start websocket thread
	#start all workers
	