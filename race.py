
from argparse import ArgumentParser

import numpy as np
import pygame

from brain import Brain
from constants import (ALL_TRACKS_BRAIN_FILE, FONT_SIZE, FPS, GENERATION_MAX_FRAMES,
                       MIN_PROGRESS_DISTANCE, POPULATION_SIZE, PROGRESS_CHECK_FRAMES,
                       SCREEN_SIZE, SHOWCASE_COUNT, SINGLE_TRACK_BRAIN_FILE, TEXT_COLOR,
                       TRAINING_TRACKS)
from evolution import next_generation
from utils import find_start_position, load_brain, load_track, save_brain

from car import Car

best_car = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--track", default="track1.png")
    parser.add_argument("--mode", choices=("manual", "NN", "autotest"), default="NN")
    # resume: start from the saved best brain and keep evolving
    # fresh:  ignore the saved brain, start from random cars
    # solo:   just watch the saved best brain drive by itself (press N to start evolving from it)
    parser.add_argument("--type", choices=("resume", "fresh", "solo"), default="resume")
    # Which brain file to START from. Leave it off to use the mode's own file
    # (NN -> single, autotest -> all). Saving always goes to the mode's own file.
    parser.add_argument("--brain", choices=("single", "all"))
    return parser.parse_args()


def drive_with_keyboard(car, track):
    """Turn key presses into car actions. The network presses the same buttons."""
    keys = pygame.key.get_pressed()
    car.rotate(left=keys[pygame.K_LEFT], right=keys[pygame.K_RIGHT])
    if keys[pygame.K_DOWN]:
        car.brake()
    if keys[pygame.K_UP]:
        car.move_forward(track)
    else:
        car.reduce_speed(track)


def draw_frame(screen, font, track, cars, show_sensors, mode, generation, frame,
               track_label, showcase):
    """Redraw the whole scene from scratch. Called once per frame; only draws, never changes state."""
    global best_car
    screen.blit(track, (0, 0))

    alive_cars = [car for car in cars if car.alive]
    # The best car is picked from the living ones. If everyone is dead, use all cars
    # so the status text still has a car to read from.
    best = max(alive_cars or cars, key=lambda car: car.fitness())
    best_car = best
    for car in cars:
        if car is not best:
            car.draw(screen)
    best.draw(screen, track, show_sensors, is_best=True)

    alive = sum(car.alive for car in cars)
    if showcase:
        status = (f"SHOWCASE - {len(cars)} car(s), no time limit   Alive: {alive}/{len(cars)}   "
                  f"(N = next generation)")
    elif mode == "manual":
        status = f"Speed: {best.vel:.1f}"
        if not best.alive:
            status += "   CRASHED - R to retry, N for a new brain"
    else:
        seconds_left = (GENERATION_MAX_FRAMES - frame) / FPS
        status = (f"Generation {generation}{track_label}   Alive: {alive}/{len(cars)}   "
                  f"Best: {best.fitness():.0f}   Time left: {seconds_left:.1f}s")
    screen.blit(font.render(status, True, TEXT_COLOR), (10, 10))

    pygame.display.flip()


def kill_cars_without_progress(cars):
    """Cars still near their spawn point (e.g. spinning in circles) are taken out."""
    for car in cars:
        if car.distance_from_start() < MIN_PROGRESS_DISTANCE:
            car.alive = False


def main():
    args = parse_args()
    autotest = args.mode == "autotest"
    brain_files = {"single": SINGLE_TRACK_BRAIN_FILE, "all": ALL_TRACKS_BRAIN_FILE}
    save_file = ALL_TRACKS_BRAIN_FILE if autotest else SINGLE_TRACK_BRAIN_FILE
    load_file = brain_files[args.brain] if args.brain else save_file

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    font = pygame.font.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    # autotest mode cycles through every track; the other modes use just one.
    track_files = TRAINING_TRACKS if autotest else [args.track]
    pygame.display.set_caption(f"{args.mode} - {', '.join(track_files)}")
    tracks = [load_track(name) for name in track_files]
    starts = [find_start_position(t) for t in tracks]    # [(start_pos, start_heading), ...]
    track_index = 0
    track = tracks[0]
    start_pos, start_heading = starts[0]

    showcase = False        # S: top cars drive with no time limit until N is pressed
    showcase_cars = []

    if args.mode == "manual":
        cars = [Car(start_pos, heading = start_heading)]
    else:
        cars = [Car(start_pos, heading = start_heading) for _ in range(POPULATION_SIZE)]
        saved = None if args.type == "fresh" else load_brain(load_file)
        if saved is not None:
            cars[0] = Car(start_pos, heading = start_heading, brain = saved)

        if args.type == "solo" and saved is not None:
            # Solo is just the showcase with one car: the saved best brain.
            showcase_cars = [cars[0]]
            showcase = True

    show_sensors = True
    generation_number = 1
    frame = 0

    running = True
    while running:
        clock.tick(FPS)
        evolve_now = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Only NN mode saves on quit. Autotest already saves the elite every
                # generation (a mid-generation car may not have been tested on every
                # track yet), and manual/solo have nothing new to save.
                if args.mode == "NN" and args.type != "solo":
                    save_brain(best_car, save_file)
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    for car in (showcase_cars if showcase else cars):
                        car.reset(start_pos, start_heading)
                    frame = 0
                elif event.key == pygame.K_e:
                    running = False   
                elif event.key == pygame.K_n:
                    evolve_now = True
                elif event.key == pygame.K_k:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_s and args.mode != "manual":
                    ranked = sorted(cars, key=lambda car: car.fitness(), reverse=True)
                    showcase_cars = [Car(start_pos, heading = start_heading, brain = car.brain)
                                     for car in ranked[:SHOWCASE_COUNT]]
                    showcase = True

        if showcase:
            # No timer, no progress check: just let the best cars drive.
            for car in showcase_cars:
                car.drive_with_network(track)
        elif args.mode != "manual":
            for car in cars:
                car.drive_with_network(track)
            frame += 1
            if frame == PROGRESS_CHECK_FRAMES:
                kill_cars_without_progress(cars)

            round_over = all(not car.alive for car in cars) or frame >= GENERATION_MAX_FRAMES
            if round_over:
                if autotest and track_index < len(tracks) - 1:
                    # Same cars, next track. Bank this track's score first.
                    for car in cars:
                        car.total_fitness += car.distance_travelled
                    track_index += 1
                    track = tracks[track_index]
                    start_pos, start_heading = starts[track_index]
                    for car in cars:
                        car.reset(start_pos, start_heading)
                    frame = 0
                else:
                    evolve_now = True
        elif cars[0].alive:
            drive_with_keyboard(cars[0], track)

        if evolve_now:
            # New generations always start on the first track.
            showcase = False
            track_index = 0
            track = tracks[0]
            start_pos, start_heading = starts[0]
            cars = next_generation(cars, start_pos, start_heading)
            if autotest:
                save_brain(cars[0], save_file)   # cars[0] is the elite (best of last generation)
            generation_number += 1
            frame = 0

        track_label = f"   Track {track_index + 1}/{len(tracks)}" if autotest else ""
        draw_frame(screen, font, track, showcase_cars if showcase else cars, show_sensors,
                   args.mode, generation_number, frame, track_label, showcase)

    pygame.quit()


if __name__ == "__main__":
    main()
