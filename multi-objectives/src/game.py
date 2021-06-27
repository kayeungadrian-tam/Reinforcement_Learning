import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import pyglet
from pyglet import clock
from time import sleep

from gym.envs.classic_control import rendering

import matplotlib.pyplot as plt

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        
        self.label.draw()

class Key():
    def __init__(self):
        self.a = 1

    def key_press(self, k, mod):
        if k == key.LEFT:
            self.a = 2

        if k == key.RIGHT:
            self.a = 0

        if k == key.DOWN:
            game.close()

        return self.a

    def key_release(self, k, mod):
        if k == key.LEFT:# and a == 0:
            self.a = 1


        if k == key.RIGHT:# and a == 2:
            self.a = 1

        return self.a

class DrawMap():
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 6))
        plt.ylim(0, 600)
        plt.xlim(0, 800)

    def draw_line(self):
        array = []
        ax = plt.gca()
        xy = plt.ginput(10)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x, y)
        ax.figure.canvas.draw()
        array.append((x,y))
        return x,y 

    def run(self, num_points):
        for _ in range(1):
            x, y = self.draw_line()
            test = []
            for u in range(num_points):
                # print(f'[{x[u]},{y[u]}]')
                test.append([x[u],y[u]])
            # print(test)

        return test

class Main(gym.Env):
    def __init__(self, myline):
        self.viewer = None  

        self.reset()

        self.goal_x = 400
        self.goal_y = 500

        self.echo_length = 50

        self.radius = 250
        self.right = 550
        self.left = 250

        self.custom_line = myline

        self.w = np.array((1,1,1))


        self.outer = self.draw_line_top(lane_width=0)
        self.inner = self.draw_line_top(lane_width=100)

        self.vector_1 = self.create_collision_vectors(self.outer)
        self.vector_2 = self.create_collision_vectors(self.inner)

    def reset(self):
        self.clock = 0
        self.x = 400
        self.y = 100 #65
        self.vel_x = 0
        self.vel_y = 0
        self.ang = 0
        # self.update_echo_vectors()

    def draw_line_top(self, lane_width):
        x_left = np.linspace(0 + lane_width,self.radius,100)
        x_right = np.linspace(self.right,self.right+self.radius - lane_width,100)

        x_left_bot = np.linspace(self.radius ,0 + lane_width,100)
        x_right_bot = np.linspace(self.right+self.radius - lane_width, self.right, 100)


        top_left = np.sqrt((self.radius - lane_width)**2-(x_left-self.left)**2)+300
        top_middle = [(self.left, 550 - lane_width),(self.right,550 - lane_width)]
        top_right = np.sqrt((self.radius - lane_width)**2-(x_right-self.right)**2)+300

        bottom_right = -np.sqrt((self.radius - lane_width)**2-(x_right_bot-self.right)**2)+300
        bottom_middle = [(self.left, 50+  lane_width),(self.right, 50 + lane_width)]
        bottom_left = -np.sqrt((self.radius - lane_width)**2-(x_left_bot-self.left)**2)+300

        line_combine = list(zip(x_left, top_left)) + top_middle + list(zip(x_right, top_right)) \
            + list(zip(x_right_bot, bottom_right)) + bottom_middle + list(zip(x_left_bot, bottom_left)) 


        return  line_combine   

    def step(self, action):
        self.clock += 1

        def on_screen():
            if self.x < 20:
                self.x = 20
            elif self.x > 780:
                self.x = 780
            if self.y < 20:
                self.y = 20
            elif self.y > 580:
                self.y = 580

        on_screen()

        self.ang += (0.0015*(action-1))*180/np.pi

        if self.ang > np.pi:
            self.ang = self.ang - 2*np.pi
        if self.ang < -np.pi:
            self.ang = self.ang + 2*np.pi

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        self.goal_ang = np.arctan2(dy,dx)

        self.goal_ang_diff = self.ang - self.goal_ang

        self.goal_ang_diff = self.goal_ang_diff/(2*np.pi)



        # print(f'self: {self.ang:.2f}\tgpal: {self.goal_ang:.2f}\tDiff:{self.ang-self.goal_ang:.2f}')

        # if self.clock < 1500:
        self.vel_x = 0.5*self.vel_x + 0.9*np.cos(self.ang)
        self.vel_y = 0.5*self.vel_y + 0.9*np.sin(self.ang)
      

        x_0, y_0 = self.x, self.y

        self.x += self.vel_x 
        self.y += self.vel_y 


        self.echo_1 = [x_0, y_0, self.x + self.echo_length*np.cos(self.ang+np.pi/4) , self.y + self.echo_length*np.sin(self.ang+np.pi/4)]
        self.echo_2 = [x_0, y_0, self.x + self.echo_length*np.cos(self.ang), self.y + self.echo_length*np.sin(self.ang)]
        self.echo_3 = [x_0, y_0, self.x + self.echo_length*np.cos(self.ang-np.pi/4) , self.y + self.echo_length*np.sin(self.ang-np.pi/4)]

        self.movement_vector = [x_0, y_0, self.x, self.y ]

        self.check_collision_env()
    
        self.reward = -3 + int(sum( self.echo_collision_distances_interp)) - 2*np.abs(self.goal_ang_diff)

        self.state = self.echo_collision_distances_interp


        self.state = np.concatenate((self.state, np.array([self.goal_ang_diff])))

        return self.state, self.reward

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 600
      
        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)


            tmp_line = rendering.make_polyline(
                self.custom_line
            )

            self.viewer.add_geom(tmp_line)

            outer = rendering.make_polyline(self.outer)    
            outer.set_color(1,0,0)
            outer.set_linewidth(3)
            self.viewer.add_geom(outer)

            inner = rendering.make_polyline(self.inner)    
            inner.set_color(1,0,0)
            inner.set_linewidth(3)
            self.viewer.add_geom(inner)

            car = rendering.make_circle(20,30, filled=False)
            car.set_color(0,0,1)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)



  
            car_line = rendering.Line((0,0),(0,self.echo_length))
            car_line.set_color(0,0,1)
            self.linetrans = rendering.Transform()
            car_line.add_attr(self.linetrans)
            self.viewer.add_geom(car_line)
  
            car_line2 = rendering.Line((0,0),(self.echo_length*np.cos(np.pi/4),self.echo_length*np.sin(np.pi/4)))
            car_line2.set_color(0.5,0.5,0.5)
            self.linetrans2 = rendering.Transform()
            car_line2.add_attr(self.linetrans2)
            self.viewer.add_geom(car_line2)

            car_line3 = rendering.Line((0,0),(self.echo_length*np.cos(3*np.pi/4),self.echo_length*np.sin(3*np.pi/4)))
            car_line3.set_color(0.5,0.5,0.5)
            self.linetrans3 = rendering.Transform()
            car_line3.add_attr(self.linetrans3)
            self.viewer.add_geom(car_line3)


            flagx = self.goal_x
            flagy1 = self.goal_y
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)


            flagx = 400
            flagy1 = 100
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, 0., 0.8)
            self.viewer.add_geom(flag)


        self.cartrans.set_translation(self.x, self.y)

        self.linetrans.set_translation(self.x, self.y)
        self.linetrans.set_rotation(self.ang-np.pi/2)

        self.linetrans2.set_translation(self.x, self.y)
        self.linetrans2.set_rotation(self.ang-np.pi/2)

        self.linetrans3.set_translation(self.x, self.y)
        self.linetrans3.set_rotation(self.ang-np.pi/2)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def display_info(self):
        text = 'Test %s' %self.x
        label = pyglet.text.Label(text, font_size=12,
                                x=10, y=10, anchor_x='left', anchor_y='bottom',
                                color=(255, 123, 255, 255))
        label.draw()
        self.viewer.add_geom(DrawText(label))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def create_collision_vectors(self, line):
        vector = np.array(line)
        n = vector.shape[0]
        line_combine = np.zeros((n-1,4))

        line_combine[:,[0,1]] = vector[:n-1,[0,1]]
        line_combine[:,[2,3]] = vector[1:,[0,1]]

        return line_combine

    def check_collision_env(self):
        max_distance = 2*self.echo_length
        distances = np.full((3), max_distance)

        vector = [self.vector_1, self.vector_2]
        for i in range(2):
            for line in vector[i]:
                result = line_intersect(*self.movement_vector, *line)
                if result is not None:
                    self.reset()
                    break
            for line2 in (vector[i]):
                result0 = line_intersect(*self.echo_1, *line2)
                if result0 is not None:
                    found = True
                    distances[0] = np.sqrt((self.x-result0[0])**2+(self.y-result0[1])**2)
                result1 = line_intersect(*self.echo_2, *line2)
                if result1 is not None:
                    found = True
                    distances[1] = np.sqrt((self.x-result1[0])**2+(self.y-result1[1])**2)
                result2 = line_intersect(*self.echo_3, *line2)
                if result2 is not None:
                    found = True
                    distances[2] = np.sqrt((self.x-result2[0])**2+(self.y-result2[1])**2)
        self.echo_collision_distances_interp = np.interp(distances, [0, max_distance], [-1, 1])

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

if __name__=='__main__':
    from pyglet.window import Window
    from pyglet.window import key

    myline = [[0,0],[10,10]]

    game = Main(myline)
    game.step(action=1)
    game.render()
    press = Key()
    game.viewer.window.on_key_press = press.key_press
    game.viewer.window.on_key_release = press.key_release

    t = 0
    while True:

        t += 1
        game.step(action = press.a)
        game.render()
