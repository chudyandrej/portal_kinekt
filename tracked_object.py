from __future__ import division
from collections import deque
import random
from collections import namedtuple
import math

MAX_HISTORY = 15
frame_width = 640
frame_height = 480


class TrackedObject:
    def __init__(self, x, y, time):
        self.history = deque()
        self.frames_since_start = 0
        self.update(x, y, time)
        self.frames_missing = 0
        self.start_x = x
        self.start_y = y
        self.changed_starting_pos = False
        self.id = int(255 * random.random())
        self.center_time = 0
        self.start_time = time

        self.color = (255 * random.random(), 255 * random.random(),
                      255 * random.random())

        self.particles = []

    def update(self, x, y, time):
        if len(self.history) >3:
            start_x = self.history[-2][0]
            start_y = self.history[-2][1]
            start_t = self.history[-2][2]
            last_x = self.history[-1][0]
            last_y = self.history[-1][1]
            last_t = self.history[-1][2]

            time_since_start = last_t - start_t

            v_x = (last_x - start_x) / time_since_start
            v_y = ((last_y - start_y) / time_since_start)
            self.init_particles(x,y, v_x, v_y)


        half_frame_height = frame_height / 2
        if (len(self.history) != 0):
            last_y = self.history[-1][1]
            last_time = self.history[-1][2]
            if last_y < half_frame_height:
                if y > half_frame_height:
                    self.center_time = (time + last_time) / 2
            else:
                if y < half_frame_height:
                    self.center_time = (time + last_time) / 2

        position = (x, y, time)
        if len(self.history) > MAX_HISTORY:
            self.history.popleft()
        self.history.append(position)
        self.frames_missing = 0
        self.frames_since_start += 1
        self.old_time = time

    def particles_update(self):
        if not self.particles:
            return
        new_particles = []
        for p in self.particles:
            new_x = p[0] + random.gauss(p[2], frame_width / 15)
            new_y = p[1] + random.gauss(p[3], frame_height / 50)
            # Try to use history based velocity estimation
            new_vx = random.gauss(p[2], p[2] / 20)
            new_vy = random.gauss(p[3], p[3] / 20)
            new_particles.append((new_x, new_y, new_vx, new_vy))
        self.particles = new_particles


    def _particle_likelihoods(self, measurements):
        particle_likelihoods = []
        for p in self.particles:
            closest_distance = min(list(map(lambda m: math.sqrt((m[0] - p[0]) ** 2 + (m[1] - p[1]) ** 2), measurements)))
            standard_deviation = 50
            likelihood = math.exp(-closest_distance ** 2 / (2 * standard_deviation ** 2  ))
            # TODO : add k
            particle_likelihoods.append(likelihood)
        return particle_likelihoods

    def particles_filter(self, measurements):
        new_generation = []
        new_generation_size = len(self.particles)
        likelihoods = self._particle_likelihoods(measurements)
        sum_likelihoods = sum(likelihoods)

        while len(new_generation) < new_generation_size:
            sum_so_far = likelihoods[0]
            i = 0
            random_num = random.random() * sum_likelihoods
            while sum_so_far < random_num:
                i += 1
                sum_so_far += likelihoods[i]
            new_generation.append(self.particles[i])

    def init_particles(self, x, y, vx, vy):
        self.particles = [(x, y, vx, vy) for _ in range(100)]




    def get_position(self):
        Point = namedtuple('Point', 'x y')
        if (len(self.history) == 0):
            return Point(0, 0)
        return Point(self.history[-1][0], self.history[-1][1])

    def abs_disto_obj(self, tracked_object, t):
        # calculate abosult distace from start position to prediction position
        pass_in = 0
        pass_out = 0
        distance = self.start_y - self.get_prediction(t).y
        prediction_distace = self.get_position().y - self.get_prediction(t).y
        if abs(prediction_distace) < frame_height / 3:
            if distance < 0:
                if abs(distance) > frame_height / 2:
                    pass_out += 1
            else:
                if abs(distance) > frame_height / 2:
                    pass_in += 1
        return pass_in, pass_out

    def missing(self):
        self.frames_missing += 1

        if self.frames_missing > 10 or self.frames_since_start < 2:
            return -1
        else:
            return 0

    def get_prediction(self, current_t):
        Point = namedtuple('Point', 'x y')
        if (len(self.history) == 0):
            return Point(0, 0)
        if (len(self.history) == 1):
            return Point(self.history[0][0], self.history[0][1])
        start_x = self.history[0][0]
        start_y = self.history[0][1]
        start_t = self.history[0][2]
        last_x = self.history[-1][0]
        last_y = self.history[-1][1]
        last_t = self.history[-1][2]

        time_since_start = last_t - start_t

        v_x = (last_x - start_x) / time_since_start
        v_y = ((last_y - start_y) / time_since_start)
        delta_t = current_t - last_t
        current_x = last_x + v_x * delta_t
        current_y = last_y + v_y * delta_t

        return Point(int(current_x), int(current_y))
