import numpy as np
import pygame
import gym
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

window_width, window_height = 800, 600
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # returns a (x, y) tuple or None if there is no intersection
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return None
    if not(0 <= s <= 1 and 0 <= t <= 1):
        return None
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


class Environment():
    def __init__(self, game):
        self.game = game

        self.line1_array_source = np.array(
            [[1, 599], [1, 1], [1, 1], [799, 1], [799, 1], [799, 599], [799, 599], [1, 599],[113.54838709677418, 562.3376623376623], [259.35483870967744, 519.48051948051943], [266.45161290322585, 527.2727272727273]]
            # [[123.87096774193549, 329.87012987012986], [212.90322580645162, 453.2467532467532], [353.54838709677415, 519.4805194805194], [518.7096774193549, 432.46753246753246], [579.3548387096774, 257.1428571428571],[1, 599], [1, 1], [1, 1], [799, 1], [799, 1], [799, 599], [799, 599], [1, 599], [150, 400], [300, 480]]
            )

        self.line2_array_source = np.array(
            [[611.6129032258065, 203.8961038961039], [642.5806451612904, 249.3506493506493], [642.5806451612904, 337.6623376623377], [615.483870967742, 485.71428571428567], [421.2903225806451, 537.6623376623377], [558.0645161290323, 437.6623376623377], [430.9677419354839, 429.87012987012986], [446.45161290322585, 384.41558441558436], [523.8709677419355, 290.9090909090909], [614.1935483870968, 205.1948051948052]]
            )

        self.load_level()

    def load_level(self):
        lin1, line2 = np.zeros((2,2)), np.zeros((2,2))
        line1 = self.line1_array_source.copy()
        line2 = self.line2_array_source.copy()
        self.set_level_vectors(line1, line2)
        self.generate_collision_vectors(line1, line2)
        # self.generate_objects()
        self.generate_objects_vectors()
        self.new_vectors()

    def set_level_vectors(self, line1, line2, level_collision_vectors=None):
        self.line1 = line1
        self.line2 = line2
        self.line1_list = line1.tolist()
        self.line2_list = line2.tolist()

        self.level_collision_vectors = level_collision_vectors

    def generate_collision_vectors(self,line1,line2):
        # for collision calculation, is numpy array
        # only call once to generate single line structe
        n1, n2 = line1.shape[0], line2.shape[0]
        line_combined = np.zeros((n1 + n2 - 2, 4))        
        line_combined[:n1-1,[0,1]] = line1[:n1-1,[0,1]]
        line_combined[:n1-1,[2,3]] = line1[1:n1,[0,1]]
        line_combined[n1-1:n1+n2-2,[0,1]] = line2[:n2-1,[0,1]]
        line_combined[n1-1:n1+n2-2,[2,3]] = line2[1:n2,[0,1]]
        self.level_collision_vectors = line_combined

    def generate_objects(self):
        # self.obj_x = np.random.randint(2, 500)
        # self.obj_y = np.random.randint(2, 500)   
        self.obj_img = pygame.transform.scale(pygame.image.load('../resources/space02_mercury.png'),(50,50))

    def generate_objects_vectors(self):
        self.obj_x = 200
        self.obj_y = 200
        self.size = 35
        vectors = np.array([
            [self.obj_x-self.size,self.obj_y+self.size],
            [self.obj_x+self.size,self.obj_y+self.size],
            [self.obj_x+self.size,self.obj_y-self.size],
            [self.obj_x-self.size,self.obj_y-self.size],
            [self.obj_x-self.size,self.obj_y+self.size]
            ])
        self.obj_list = vectors.tolist()
        n = vectors.shape[0]
        obj_line = np.zeros((n-1,4))
        obj_line[:, [0,1]] = vectors[:n-1, [0,1]]
        obj_line[:, [2,3]] = vectors[1:, [0,1]]
        self.object_vectors = obj_line
        self.new_vectors()


    def new_vectors(self):
        self.size += 10
        vectors2 = np.array([
            [self.obj_x-self.size,self.obj_y+self.size],
            [self.obj_x+self.size,self.obj_y+self.size],
            [self.obj_x+self.size,self.obj_y-self.size],
            [self.obj_x-self.size,self.obj_y-self.size],
            [self.obj_x-self.size,self.obj_y+self.size]
            ])
        self.new_list = vectors2.tolist()
        n = vectors2.shape[0]
        obj_line2 = np.zeros((n-1,4))
        obj_line2[:, [0,1]] = vectors2[:n-1, [0,1]]
        obj_line2[:, [2,3]] = vectors2[1:, [0,1]]
        self.new_obj_vectors = obj_line2

class Rocket():
    n_echo = 3
    rotation_max = 0.15
    def __init__(self, game):
        self.game = game
        self.env = Environment(self)
        # self.env.generate_objects()
        self.reset_game_state()
        self.update_echo_vectors()

    def reset_game_state(self, x=50, y=50, ang=-0.8, vel_x=0, vel_y=0):
        self.update_state(np.array([x, y, ang, vel_x, vel_y]))
        self.done = False
        self.action = 0
        self.action_state = 0
        self.framecount = 0

    def update_state(self, rocket_state):
        self.x = rocket_state[0]
        self.y = rocket_state[1]
        self.ang = rocket_state[2]
        self.vel_x = rocket_state[3]
        self.vel_y = rocket_state[4]

    def accelerate(self, accelerate):  # input: action0
        # backwards at half speed
        accelerate = accelerate * 0.5
        if accelerate > 0:
            accelerate = accelerate * 0.3

        # * velocity in abhängigkeit von ANG, also raketen stellung
        self.vel_x = 0.95*self.vel_x + accelerate * np.cos(self.ang) 
        self.vel_y = 0.95*self.vel_y - accelerate * np.sin(self.ang) 

    def rotate(self, rotate):  # input: action1
        self.ang = self.ang + self.rotation_max * rotate
        # get angular in range of -pi,pi
        if self.ang > np.pi:
            self.ang = self.ang - 2 * np.pi
        if self.ang < -np.pi:
            self.ang = self.ang + 2 * np.pi

    def move(self, action):
        self.rotate(action)
        self.accelerate(1.)
        # displacement
        d_x, d_y = self.vel_x, self.vel_y
        x_from, y_from = self.x, self.y

        self.x = self.x + d_x
        self.y = self.y + d_y
        self.movement_vector = [x_from, y_from, self.x, self.y]

        # keep rocket on screen (optional)
        if self.x > window_width:
            self.x = window_width #self.x - window_width
            self.vel_x = 0
        elif self.x < 0:
            self.x = 0 #self.x + window_width
            self.vel_x = 0
        if self.y > window_height:
            self.y = window_height #self.y - window_height
            self.vel_y = 0
        elif self.y < 0:
            self.y = 0 #self.y + window_height
            self.vel_y = 0

    def update_echo_vectors(self):
        n = self.n_echo
        n_sideangles = int((n-1)/2)
        matrix = np.zeros((n, 4))
        matrix[:, 0] = int(self.x)
        matrix[:, 1] = int(self.y)
        # straight angle
        matrix[n_sideangles, 2] = int(self.x + 100 * np.cos(self.ang))
        matrix[n_sideangles, 3] = int(self.y - 100 * np.sin(self.ang))
        # angles from 90 deg to 0
        # ignore first angle
        angles = np.linspace(0, np.pi/6, n_sideangles+1)
        for i in range(n_sideangles):
            # first side
            matrix[i, 2] = int(self.x + 100 * np.cos(self.ang + angles[i+1]))  # x2
            matrix[i, 3] = int(self.y - 100 * np.sin(self.ang + angles[i+1]))  # y2
            # second side
            matrix[-(i+1), 2] = int(self.x + 100 * np.cos(self.ang - angles[i+1]))  # x2
            matrix[-(i+1), 3] = int(self.y - 100 * np.sin(self.ang - angles[i+1]))  # y2
        self.echo_vectors = matrix

    def check_collision_echo(self):
        max_distance = 1500 
        n = self.env.level_collision_vectors.shape[0]
        distances = np.full((self.n_echo), max_distance) # distances for observation
        for i in range(self.n_echo):
            found = False
            line1 = self.echo_vectors[i, :]
            distances_candidates = np.full((n), max_distance)
            for j, line2 in enumerate(self.env.level_collision_vectors):
                result = line_intersect(*line1, *line2)
                if result is not None:
                    found = True
                    distances_candidates[j] = np.sqrt((self.x-result[0])**2+(self.y-result[1])**2)
            for k, line3 in enumerate(self.env.object_vectors):
                result = line_intersect(*line1, *line3)
                if result is not None:
                    found = True
                    distances_candidates[j] = np.sqrt((self.x-result[0])**2+(self.y-result[1])**2)
            if found:
                argmin = np.argmin(distances_candidates) 
                distances[i] = distances_candidates[argmin]
        self.echo_collision_distances_interp = np.interp(distances, [0, 1000], [-1, 1])

    def check_collision_env(self):
        for line in self.env.level_collision_vectors:
            result = line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.action_state = 3
                self.game.set_done()
                break
            else:
                self.action_state = 1
        for line in self.env.object_vectors:
            result = line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.action_state = 3
                self.game.set_done()
                break
            else:
                self.action_state = 1
        
        self.env.new_vectors()

        for line in self.env.new_obj_vectors:
            result = line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.action_state = 3
                self.game.set_done()
                break
            else:
                self.action_state = 1

class Main(gym.Env):
    def __init__(self):
        self.reset()

    def render(self):
        import os
        def init_render(self):
            self.window = pygame.display.set_mode((window_width, window_height))
            self.window.fill((255,255,255))
            self.clock = pygame.time.Clock()
            self.ROCKET_IMG = [
                pygame.transform.scale(pygame.image.load(os.path.join('../resources', 'rocket_no_power.png')), (40,25)),
                pygame.transform.scale(pygame.image.load(os.path.join('../resources', 'rocket_power.png')), (40,25)), 
                pygame.transform.scale(pygame.image.load(os.path.join('../resources', 'rocket_power_front.png')), (40,25)),
                pygame.transform.scale(pygame.image.load(os.path.join('../resources', 'rocket_black.png')), (40,25))
                ]
            pygame.init()

        def draw_level():
            pygame.draw.lines(self.window, (0, 0, 0), False, self.env.line1_list, 4)
            pygame.draw.lines(self.window, (0, 0, 0), False, self.env.line2_list, 4)

        def draw_rocket():
            self.rocket.img = self.ROCKET_IMG[self.rocket.action_state]
            rotated_image = pygame.transform.rotate(self.rocket.img, self.rocket.ang / np.pi * 180)
            new_rect = rotated_image.get_rect(center=self.rocket.img.get_rect(
                center=(self.rocket.x, self.rocket.y)).center)
            self.window.blit(rotated_image, new_rect.topleft)

        def draw_echo_vectors():
            n = self.rocket.n_echo
            echo_vectors_short = self.rocket.echo_vectors
            # if len(self.rocket.echo_collision_points) == n:
            #     echo_vectors_short = self.rocket.echo_vectors
            #     for i in range(n):
            #         echo_vectors_short[i,[2,3]] = self.rocket.echo_collision_points[i]     
            for vector in echo_vectors_short:
                pygame.draw.line(self.window, (255, 40, 40), vector[0:2], vector[2:4], 1)

        def draw_objects(): 
            pygame.draw.lines(self.window, (0, 255, 0), False, self.env.obj_list, 4)
            pygame.draw.lines(self.window, (0, 255, 255), False, self.env.new_list, 4)
            

        def display_ui():
            myfont = pygame.font.SysFont('Segoe UI', 15)
            display_loc = myfont.render(f'x: {self.rocket.x:.1f}   y:{self.rocket.y:.1f}', True, (0,0,0))
            display_rocket = myfont.render(f'  v: {np.sqrt(self.rocket.vel_x**2 + self.rocket.vel_y**2):.2f}    angle: {self.rocket.ang*180/np.pi:.2f}', True, (0,0,0))
            display_frame = myfont.render(f'frame: {self.rocket.framecount}', True, (0,0,0))
            # display_goal = myfont.render(f'Distance to Goal: {np.sqrt((self.rocket.x-self.food.x_food)**2 + (self.rocket.y-self.food.y_food)**2):.2f} ', True, (0,0,0))
            self.window.blit(display_loc, (5,1))
            self.window.blit(display_rocket, (120,1))
            self.window.blit(display_frame, (720,1))
            # self.window.blit(display_goal, (400,1))
        
        init_render(self)
        draw_objects()
        draw_echo_vectors()
        draw_rocket()
        display_ui()
        draw_level()


        pygame.display.update()

    def set_done(self):
        self.rocket.done = True
        self.reward = -500
        self.rocket.reset_game_state()

    def step(self, action=0):
        action = max(min(action,1),-1)

        self.rocket.move(action)
        self.rocket.update_echo_vectors()

        self.rocket.check_collision_env()   
        self.rocket.check_collision_echo()

        distances = self.rocket.echo_collision_distances_interp

        if not self.rocket.done:
            self.reward = -3 + int(sum(distances))

        self.rocket.framecount += 1
        self.steps += 1
        return self.reward, distances

    def reset(self):
        self.env = Environment(self)
        self.rocket = Rocket(self)
        self.score = 0
        self.steps = 0

def pressed_to_action(keytouple):
    action_turn = 0.
    if keytouple[K_LEFT]:
        action_turn += 1
    if keytouple[K_RIGHT]:
        action_turn -= 1   

    return action_turn


if __name__=="__main__":

    game = Main()
    game.render()
    run = True
    while run:
        game.clock.tick(30)
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    run = False
            elif event.type == pygame.QUIT:
                run = False
        get_pressed = pygame.key.get_pressed()
        action = pressed_to_action(get_pressed)
        game.step(action)
        game.render()
    pygame.quit()