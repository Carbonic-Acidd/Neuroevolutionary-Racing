
import math

import numpy as np
import pygame

from brain import Brain
from constants import (CAR_ACCELERATION, CAR_MAX_VEL, CAR_ROTATION_VEL, CAR_BASE_START_HEADING,
                       CORNER_COLOR_ALIVE, CORNER_COLOR_DEAD, IMAGES_DIR, N_INPUTS, N_OUTPUTS,
                       SENSOR_ANGLES, SENSOR_COLOR, SENSOR_MAX_RANGE, SENSOR_STEP,
                       STILL_LIMIT_FRAMES)
from network import forward_pass
from utils import blit_rotate_center, cast_ray, get_rotated_corners, is_grass

class Car:
    """
    heading is the direction the car faces, in degrees, using normal screen angles:
        0 -> RIGHT, 90 -> DOWN, 180 -> LEFT, 270 -> UP
    (y grows downward on screen, so positive angles turn clockwise.)
    This is the same convention cast_ray uses, so no conversions are needed anywhere.
    """

    IMAGES = None   # {"normal": Surface, "best": Surface, "dead": Surface}, loaded once

    def __init__(self, location, brain=None, max_vel=CAR_MAX_VEL, rotation_vel=CAR_ROTATION_VEL,
                 heading=CAR_BASE_START_HEADING):
        if Car.IMAGES is None:
            # Load from disk exactly once for the whole program, shared by every car.
            # The PNGs face down; rotate them once so they face right, matching heading 0.
            Car.IMAGES = {
                name: pygame.transform.rotate(
                    pygame.image.load(f"{IMAGES_DIR}{name}_car.png").convert_alpha(), 90)
                for name in ("normal", "best", "dead")
            }
        self.image = Car.IMAGES["normal"]   # used for the hitbox size
        self.half_w = self.image.get_width() / 2
        self.half_h = self.image.get_height() / 2
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.acceleration = CAR_ACCELERATION

        # Step 4: the brain. It lives in __init__, not reset(), so resetting the
        # car's position keeps the same brain.
        self.sensor_values = np.zeros(N_INPUTS)
        #self.hidden_values = [0.0] * N_HIDDEN
        self.output_values = [0.0] * N_OUTPUTS
        self.brain = brain if brain is not None else Brain()

        self.distance_travelled = 0
        self.frames_still = 0
        # Score banked from earlier tracks (autotest mode). Lives here, not in reset(),
        # so it carries over when the car moves to the next track.
        self.total_fitness = 0

        self.reset(location, heading)

    def reset(self, location, heading):
        self.x, self.y = location
        self.start_x, self.start_y = location
        self.heading = heading
        self.vel = 0
        self.alive = True
        self.distance_travelled = 0
        self.frames_still = 0

    # ---- controls (later the neural network will call these) ----

    def rotate(self, left=False, right=False):
        if left:
            self.heading -= self.rotation_vel
        elif right:
            self.heading += self.rotation_vel
        self.heading %= 360

    def move_forward(self, track):
        if not self.alive:
            return
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        self.check_crash(track)

    def reduce_speed(self, track):
        """Coasting: no throttle, so the car slowly loses speed."""
        if not self.alive:
            return
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()
        self.check_crash(track)

    def brake(self):
        if self.alive:
            self.vel = max(self.vel - self.acceleration * 2, 0)
    

    # ---- Step 4: let the network drive ----

    def drive_with_network(self, track):
        """
        sensors -> brain -> two numbers -> the same buttons the keyboard presses.
        steering: below -0.1 = left, above 0.1 = right, in between = straight
        throttle: above 0 = accelerate, otherwise coast
        """
        if not self.alive:
            return
        
        self.sensor_values = self.get_sensor_distances(track)
        self.output_values = forward_pass(self.sensor_values, self.brain)
        steering, throttle = self.output_values

        self.heading = (self.heading + steering * self.rotation_vel) % 360
        if throttle > 0:
            self.move_forward(track)
        else:
            self.reduce_speed(track)

    # ---- physics and collision ----

    def move(self):
        radians = math.radians(self.heading)
        self.x += math.cos(radians) * self.vel
        self.y += math.sin(radians) * self.vel
        self.distance_travelled += self.vel
        if self.vel < 0.01:
            self.frames_still +=1
            if self.frames_still > STILL_LIMIT_FRAMES:
                self.alive = False
        else:
            self.frames_still = 0

    def get_corners(self):
        return get_rotated_corners(self.x, self.y, self.half_w, self.half_h, self.heading)

    def check_crash(self, track):
        if self.alive and any(is_grass(track, x, y) for x, y in self.get_corners()):
            self.alive = False
            self.vel = 0

    # ---- sensors ----

    def get_sensor_readings(self, track):
        """Cast one ray per sensor angle. Returns [(distance, end_point), ...]."""
        return [cast_ray(track, self.x, self.y, self.heading + offset, SENSOR_MAX_RANGE, step=SENSOR_STEP)
                for offset in SENSOR_ANGLES]

    def get_sensor_distances(self, track):
        """
        The network's inputs: five numbers between 0 (touching grass) and 1 (far away).
        Scaling to 0-1 matters: networks learn much better with small, consistent inputs.
        """
        return np.asarray([distance / SENSOR_MAX_RANGE for distance, _ in self.get_sensor_readings(track)])

    def draw(self, screen, track=None, show_sensors=False, is_best=False):
        # Pick the sprite: dead beats best, best beats normal. This is just a dict lookup.
        if not self.alive:
            image = Car.IMAGES["dead"]
        elif is_best:
            image = Car.IMAGES["best"]
        else:
            image = Car.IMAGES["normal"]
        # pygame rotates counter-clockwise, heading turns clockwise, hence the minus
        blit_rotate_center(screen, image, (self.x, self.y), -self.heading)
        color = CORNER_COLOR_ALIVE if self.alive else CORNER_COLOR_DEAD
        for cx, cy in self.get_corners():
            pygame.draw.circle(screen, color, (int(cx), int(cy)), 3)
        if show_sensors and track is not None:
            for _, (end_x, end_y) in self.get_sensor_readings(track):
                pygame.draw.line(screen, SENSOR_COLOR, (self.x, self.y), (end_x, end_y), 1)
                pygame.draw.circle(screen, SENSOR_COLOR, (int(end_x), int(end_y)), 3)

    def distance_from_start(self):
        """Straight-line distance from where this car spawned. Spinning cars keep this small."""
        return math.hypot(self.x - self.start_x, self.y - self.start_y)

    def fitness(self):
        return self.total_fitness + self.distance_travelled