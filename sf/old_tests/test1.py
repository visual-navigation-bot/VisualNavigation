from simulation import *
import pygame
import numpy as np
import random

# Constants
PARTY_V = 50 * np.array([[0., 2.1, 2.1, 2.1], [2.1, 0., 2.1, 2.1], [2.1, 2.1, 0., 2.1], [2.1, 2.1, 2.1, 0.]])
PARTY_R = 0.5 * np.array([[0., 100., 100., 100.], [100., 0., 100., 100.], [100., 100., 0., 100.], [100., 100., 100., 0.]])

# Generate Pygame Simulation Environment
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Pedestrian Simulation')
screen.fill((255, 255, 255))
clock = pygame.time.Clock()

fps = 50.
tdiff = 1 / fps

# setup obstacles
obstacles = []
start = np.array([0, 100])
end = np.array([800,100])
obstacles.append(Obstacle(start, end))
start = np.array([0, 500])
end = np.array([800, 500])
obstacles.append(Obstacle(start, end))



# setup pedestrians
pedestrians = []
init_s = np.array([100., 150.])
exp_s = np.array([900., 150.])
v0 = 100.
init_v = np.array([100., 0])
tau = 0.5
U = 100.
R = 500.
p = Ped(init_s, exp_s, v0, init_v, tau, U, R)
pedestrians.append(p)

init_s = np.array([100., 200.])
exp_s = np.array([900., 200.])
v0 = 100.
init_v = np.array([100., 0])
tau = 0.5
U = 100.
R = 500.
p = Ped(init_s, exp_s, v0, init_v, tau, U, R)
pedestrians.append(p)

init_s = np.array([100., 250.])
exp_s = np.array([900., 250.])
v0 = 100.
init_v = np.array([100., 0])
tau = 0.5
U = 100.
R = 500.
p = Ped(init_s, exp_s, v0, init_v, tau, U, R)
pedestrians.append(p)

init_s = np.array([100., 300.])
exp_s = np.array([900., 300.])
v0 = 100.
init_v = np.array([100., 0])
tau = 0.5
U = 100.
R = 500.
p = Ped(init_s, exp_s, v0, init_v, tau, U, R)
pedestrians.append(p)
ped_num = len(pedestrians)


running = True
while running:
    clock.tick(fps)
    screen.fill((255,255,255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # display obstacles
    for obs in obstacles:
        obs.display()

    # relative distance matrix to all pedestians
    s = pedestrians[0].s
    repeat = np.expand_dims(np.tile(s, (ped_num,1)), axis = 0)
    for pid in range(1, ped_num):
        s = pedestrians[pid].s
        r = np.expand_dims(np.tile(s, (ped_num,1)), axis = 0)
        repeat = np.concatenate((repeat, r))
    dist_matr = np.swapaxes(repeat, 0, 1) - repeat
    

    # update pedestrians and display them
    for pid in range(ped_num):
        # relative information to other pedestrians
        other_dis = np.delete(dist_matr[pid], pid, 0)
        other_V = np.delete(PARTY_V[pid], pid, 0)
        other_R = np.delete(PARTY_R[pid], pid, 0)

        pedestrians[pid].move(other_dis, other_V, other_R, tdiff)
        pedestrians[pid].display()

    pygame.display.flip()