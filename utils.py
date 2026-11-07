import math

import numpy as np
import pygame

from constants import (GRASS, GRASS_TOLERANCE, IMAGES_DIR, SCREEN_SIZE,
                       START_COLOR, START_TOLERANCE)

def load_track(path):
    """Load a track image and stretch it to the screen size."""
    track = pygame.image.load(IMAGES_DIR + path).convert()
    # transform.scale (not smoothscale) keeps colors pure instead of blending edges
    return pygame.transform.scale(track, SCREEN_SIZE)


def color_matches(color, target, tolerance):
    """True if every R, G, B value is within `tolerance` of the target."""
    return all(abs(c - t) <= tolerance for c, t in zip(tuple(color)[:3], target))


def is_grass(track, x, y):
    # Leaving the screen counts as grass, so cars can't escape
    if x < 0 or y < 0 or x >= track.get_width() or y >= track.get_height():
        return True
    return color_matches(track.get_at((int(x), int(y))), GRASS, GRASS_TOLERANCE)


def is_start(track, x, y):
    if x < 0 or y < 0 or x >= track.get_width() or y >= track.get_height():
        return False
    return color_matches(track.get_at((int(x), int(y))), START_COLOR, START_TOLERANCE)


def find_start_position(track):
    """
    Find every white pixel and return their average position.
    That's the middle of the start line, which is where cars will spawn.

    The original loops over pixels one at a time. numpy checks them all at once,
    which is the same idea but much faster.
    """
    pixels = pygame.surfarray.array3d(track).astype(int)   # shape: (width, height, 3)
    close_enough = np.abs(pixels - START_COLOR) <= START_TOLERANCE
    is_white = close_enough.all(axis=2)                   # True where R, G and B all match
    xs, ys = np.nonzero(is_white)
    if len(xs) == 0:
        raise ValueError("No start line found. Paint a white line across the road.")
    return int(xs.mean()), int(ys.mean())


# ----------------------------------------------------------------------
# Step 2: the car
# ----------------------------------------------------------------------

def get_rotated_corners(cx, cy, half_w, half_h, angle_degrees):
    """
    The four corners of a rectangle centred at (cx, cy), rotated by angle_degrees.
    This is a standard 2D rotation: rotate each corner offset, then add the centre.
    Order: back-left, front-left, front-right, back-right (front = +x, i.e. the heading).
    """
    radians = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    offsets = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
            for lx, ly in offsets]


def blit_rotate_center(screen, image, center, angle):
    """Rotate an image and draw it so it stays centred on the same point."""
    rotated = pygame.transform.rotate(image, angle)
    screen.blit(rotated, rotated.get_rect(center=center).topleft)


# ----------------------------------------------------------------------
# Step 3: sensors
# ----------------------------------------------------------------------

def cast_ray(track, carX, carY, angle_Deg, max_range, step):
    """
    Walk outward from (cx, cy) in small steps until we hit grass or run out of range.
    Returns (distance, end_point).

    `step` trades accuracy for speed: 4 px means up to 50 checks per ray.
    With 5 rays and 40 cars that's ~10,000 pixel checks per frame, so it adds up.
    """
    radians = math.radians(angle_Deg)
    dx, dy = math.cos(radians), math.sin(radians)
    distance = 0
    while distance < max_range:
        px, py = carX + dx * distance, carY + dy * distance
        if is_grass(track, px, py):
            return distance, (px, py)
        distance += step
    return max_range, (carX + dx * max_range, carY + dy * max_range)