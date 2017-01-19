import pygame
import time
import math
from random import randint
import random

class Particle:
    def __init__(self, (x, y), radius, speed, angle):
        self.color = (0, 0, 255)
        self.x = x
        self.y = y
        self.radius = radius
        self.thick = 1
        self.speed = speed
        self.angle = angle
    def display(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius, self.thick)
    def move(self):
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed

screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption('Test')

particles = []
for pid in range(0):
    speed = random.random()
    angle = random.uniform(0, math.pi*2)
    p = Particle((randint(0, 300), randint(0, 200)), randint(5, 30), speed, angle) 
    particles.append(p)

speed = 1
angle = math.pi
p = Particle((0,0), 10, speed, angle)
particles.append(p)

pygame.display.flip()

ti = time.time()

running = True
clock = pygame.time.Clock()
ts = time.time()

while running:
    clock.tick(50)
    te = time.time()
    print te - ti
    ti = te
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255,255,255))
    for particle in particles:
        particle.move()
        particle.display()
    pygame.display.flip()

print time.time() - ts
