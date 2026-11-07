
from argparse import ArgumentParser

import numpy as np
import pygame

from brain import Brain
from constants import (FONT_SIZE, FPS, GENERATION_MAX_FRAMES, POPULATION_SIZE,
                       SCREEN_SIZE, TEXT_COLOR)
from evolution import next_generation
from utils import find_start_position, load_brain, load_track, save_brain

from car import Car

best_car = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--track", default="track1.png")
    parser.add_argument("--mode", choices=("manual", "NN"), default="NN")
    parser.add_argument("--fresh", action = "store_true")
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


def draw_frame(screen, font, track, cars, show_sensors, mode,generation, frame):
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
    
    status = ""

    if mode == "NN":
        alive = sum(car.alive for car in cars)
        seconds_left = (GENERATION_MAX_FRAMES - frame) / FPS
        status = (f"Generation {generation}   Alive: {alive}/{len(cars)}   "
                  f"Best: {best.fitness():.0f}   Time left: {seconds_left:.1f}s")
    else:
        status = f"Speed: {best.vel:.1f}"
        if not best.alive:
            status += "   CRASHED - R to retry, N for a new brain"
    screen.blit(font.render(status, True, TEXT_COLOR), (10, 10))

    pygame.display.flip()

def main():
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(f"{args.mode} - {args.track}")
    font = pygame.font.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    track = load_track(args.track)
    start_pos, start_heading = find_start_position(track)
    if args.mode == "NN":
        cars = [Car(start_pos,heading = start_heading) for _ in range (POPULATION_SIZE)]
        if not args.fresh:
            cars[0] = Car(start_pos, heading = start_heading, brain= load_brain())
    else:
        cars = [Car(start_pos, heading = start_heading)]
    show_sensors = True
    generation_number = 1
    frame = 0


    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_brain(best_car)
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    for car in cars:
                        car.reset(start_pos, start_heading)
                    frame = 0
                elif event.key == pygame.K_n:
                    generation_number += 1
                    cars = next_generation(cars, start_pos, start_heading)
                    frame = 0
                elif event.key == pygame.K_k:
                    show_sensors = not show_sensors

        if args.mode == "NN":
            for car in cars:
                car.drive_with_network(track)
            frame +=1
            generation_over = all(not car.alive for car in cars) or frame >= GENERATION_MAX_FRAMES

            if generation_over:
                cars = next_generation(cars, start_pos, start_heading)
                generation_number += 1
                frame = 0
        elif cars[0].alive:
            drive_with_keyboard(cars[0], track)

        draw_frame(screen, font, track, cars, show_sensors, args.mode, generation_number, frame)

    pygame.quit()


if __name__ == "__main__":
    main()