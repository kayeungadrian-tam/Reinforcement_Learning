import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import pyglet
from pyglet import clock
from time import sleep

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        
        self.label.draw()

class Main(gym.Env):
    def __init__(self):
        self.viewer = None
        self.clock = 0




        self.x = 60
        self.y = 60
        self.vel_x = 0
        self.vel_y = 0
        self.ang = np.pi/2

        self.goal_x = 700
        self.goal_y = 550

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

        self.ang += (0.0015*action)*180/np.pi

        if self.ang > np.pi:
            self.ang = self.ang - 2*np.pi
        if self.ang < -np.pi:
            self.ang = self.ang + 2*np.pi

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        self.goal_ang = np.arctan2(dy,dx)

        self.goal_ang_diff = self.ang - self.goal_ang

        # print(f'self: {self.ang:.2f}\tgpal: {self.goal_ang:.2f}\tDiff:{self.ang-self.goal_ang:.2f}')

        if self.clock < 1000:
            self.vel_x = 0.5*self.vel_x + 1*np.cos(self.ang)
            self.vel_y = 0.5*self.vel_y + 1*np.sin(self.ang)

            self.x += self.vel_x 
            self.y += self.vel_y 
    

            self.reward = 0.8 + (-1/1000)*np.sqrt((self.goal_x-self.x)**2+(self.goal_y-self.y)**2) - np.abs(self.goal_ang_diff)

            self.state = np.full((3), 1000)

            s_1 = np.interp(self.x, [0,1000], [0,1])
            s_2 = np.interp(self.y, [0,1000], [0,1])

            self.state[0] = self.x/1000
            self.state[1] = self.y/1000
            self.state[2] = 2*self.goal_ang_diff    
 

            self.obs = np.concatenate((self.state, np.array([self.goal_ang_diff])))


        else:
            self.x = np.random.randint(10,700)
            self.y = np.random.randint(10,500)
            self.vel_x = 0
            self.vel_y = 0
            self.ang = np.random.uniform(0., np.pi)
            self.clock = 0

        return self.state, self.reward

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        center_x = 400
        center_y = 300

        radius = 300



        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            aline = rendering.Line((50, 50), (750, 50))  # 1
            aline.set_color(1,0,0)
            self.viewer.add_geom(aline)

            aline2 = rendering.Line((50, 550), (750, 550))  # 1
            self.viewer.add_geom(aline2)
            
            xs = np.linspace(0,800,1000)
            ys = np.sqrt(radius**2-(xs - center_x)**2) + center_y
            xys = list(zip(xs,ys))
            track = rendering.make_polyline(xys)
            self.viewer.add_geom(track)

            acircle = rendering.make_circle(50, 4, filled=False)  # 3 *Note that the translation operation is also done below
            acircle.set_color(0, 1, 0)
            acircle.set_linewidth(5)  # Set line width
            # Add a pan operation
            transform1 = rendering.Transform(translation=(200, 200))  # Relative offset
            # Let the circle add the attribute of translation
            acircle.add_attr(transform1)
            self.viewer.add_geom(acircle)


            car = rendering.make_circle(20,30, filled=False)
            car.set_color(0,0,1)

            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)


            car_line = rendering.Line((0,0),(0,20))
            car_line.set_color(0,0,1)
            self.linetrans = rendering.Transform()
            car_line.add_attr(self.linetrans)
            self.viewer.add_geom(car_line)

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






        self.cartrans.set_translation(self.x, self.y)
        self.linetrans.set_translation(self.x, self.y)
        self.linetrans.set_rotation(self.ang-np.pi/2)


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

class Key():
    def __init__(self):
        self.a = 0

    def key_press(self, k, mod):
        if k == key.LEFT:
            self.a = 1

        if k == key.RIGHT:
            self.a = -1

        if k == key.DOWN:
            game.close()

        return self.a

    def key_release(self, k, mod):
        if k == key.LEFT:# and a == 0:
            self.a = 0


        if k == key.RIGHT:# and a == 2:
            self.a = 0

        return self.a




if __name__=='__main__':
    from pyglet.window import Window
    from pyglet.window import key



    game = Main()
    game.render()
    press = Key()
    game.viewer.window.on_key_press = press.key_press
    game.viewer.window.on_key_release = press.key_release

    t = 0
    while True:

        t += 1
        game.step(action = press.a)

        game.render()
  









    print('Hello World')






