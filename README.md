# Neuroevolutionary-Racing
Neuroevolutionary Algorithm for a race car to drive around a track

## Usage
```
python race.py --mode manual  --track track2.png   # drive yourself with the arrow keys
python race.py --mode NN      --track track1.png   # evolve cars on one track (single_track_brain.npz)
python race.py --mode autotest                     # evolve on track1/2/3 each generation (all_tracks_brain.npz)
python race.py --type solo --brain all --track track3.png   # watch the all-tracks brain drive track3
```
`--type` picks how NN/autotest start (ignored in manual):
- `resume` (default): start from the saved best brain and keep evolving
- `fresh`: ignore the saved brain, start from random cars
- `solo`: watch the saved best brain drive alone; press N to start evolving from it

`--brain single|all` picks which file to start from. Leave it off to use the mode's own file
(NN uses single, autotest uses all). Saving always goes to the mode's own file, so a
single-track run never overwrites the all-tracks brain.

## Keys
| Key | Action |
|-----|--------|
| R | Reset cars to the start line |
| N | Next generation (also leaves showcase) |
| S | Showcase: top 5 cars drive with no time limit |
| K | Toggle sensor rays |

## Rules
- Each track gets 20 s (`GENERATION_MAX_TIME`).
- At 5 s, cars still within 80 px of the start (spinning or stuck) are killed.
- In autotest mode a car's fitness is its total distance over all three tracks.
