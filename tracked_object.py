
from __future__ import division
from collections import deque
import random
from collections import namedtuple
MAX_HISTORY = 15
MIN_DISTANCE_OF_PASS = 250
frame_width = 640;
frame_height = 480;
class TrackedObject():

    def __init__(self, x, y, time):
        self.history = deque()
        self.frames_since_start = 0
        self.update (x, y, time)
        self.frames_missing = 0
        self.start_x = x
        self.start_y = y
        self.changed_starting_pos = False
        self.id = int(255*random.random())

        self.color = (255*random.random(), 255*random.random(),
                255*random.random())

    def update (self, x, y, time):

        position = (x, y, time)
        if len(self.history) > MAX_HISTORY:
            self.history.popleft()
        self.history.append(position)
        self.frames_missing = 0
        self.frames_since_start += 1

    def get_position(self):
        Point = namedtuple('Point', 'x y')
        if (len(self.history) == 0) :
            return Point(0,0)
        return Point(self.history[-1][0], self.history[-1][1])


    def abs_disto_obj(self,tracked_object, t):
        #calculate abosult distace from start position to prediction position
        pass_in = 0
        pass_out = 0
        distance = self.start_y - self.get_prediction(t).y
        prediction_distace = self.get_position().y - self.get_prediction(t).y
        if abs(prediction_distace) < frame_height / 3:
            if distance < 0:    
                if abs(distance) > frame_height / 2:
                    pass_out+=1                     
            else: 
                if abs(distance) > frame_height / 2:
                    pass_in+=1
        return pass_in , pass_out



    def missing(self):
        self.frames_missing += 1

        if self.frames_missing > 10 or self.frames_since_start < 2:
            return -1
        else:
            return 0

   
    def get_prediction(self, current_t):
        Point = namedtuple('Point', 'x y')
        if (len(self.history) == 0) :
            return Point(0,0)
        if (len(self.history) == 1) :
            return Point(self.history[0][0],self.history[0][1])
        start_x = self.history[0][0]
        start_y = self.history[0][1]
        start_t = self.history[0][2]
        last_x = self.history[-1][0]
        last_y = self.history[-1][1]
        last_t = self.history[-1][2]

        time_since_start = last_t - start_t

        v_x = (last_x - start_x) / time_since_start
        v_y = ((last_y - start_y) / time_since_start ) * 1.5

        delta_t = current_t - last_t

        current_x = last_x + v_x * delta_t

        current_y = last_y + v_y * delta_t
    


        return Point(int(current_x), int(current_y))
