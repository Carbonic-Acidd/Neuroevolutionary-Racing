
from argparse import ArgumentParser

import pygame

from utils import SCREEN_SIZE, find_start_position, load_track

from car import Car

FPS = 60

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--track", default="track1.png")
    parser.add_argument("--mode", choices=("manual", "NN"), default="NN")
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


def draw_frame(screen, font, track, car, show_sensors, mode):
    """Redraw the whole scene from scratch. Called once per frame; only draws, never changes state."""
    screen.blit(track, (0, 0))
    car.draw(screen, track, show_sensors)

    status = f"Speed: {car.vel:.1f}"
    if mode == "NN":
        steering, throttle = car.output_values
        status += f"   Steering: {steering:+.2f}   Throttle: {throttle:+.2f}"
    if not car.alive:
        status += "   CRASHED - R to retry, N for a new brain"
    screen.blit(font.render(status, True, (255, 255, 255)), (10, 10))

    pygame.display.flip()


def main():
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(f"{args.mode} - {args.track}")
    font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()

    track = load_track(args.track)
    start_pos = find_start_position(track)
    car = Car(start_pos)
    show_sensors = True

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    car.reset(start_pos)
                elif event.key == pygame.K_n:
                    car = Car(start_pos)   # a brand-new car means a brand-new random brain
                elif event.key == pygame.K_k:
                    show_sensors = not show_sensors

        if car.alive:
            if args.mode == "NN":
                car.drive_with_network(track)
            else:
                drive_with_keyboard(car, track)

        draw_frame(screen, font, track, car, show_sensors, args.mode)

    pygame.quit()


if __name__ == "__main__":
    main()