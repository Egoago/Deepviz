from collections import deque

import numpy as np

from src.environment.game import objects
from src.environment.game.game import Game
from src.environment.game.render import Renderer
from src.environment.ui import UI


class View:
    def __init__(self, fps, show):
        resolution = 512
        self.aspect = 0.5
        self.stack = 1
        self.show = show
        self.renderer = Renderer((int(self.aspect * resolution), resolution),
                                 fps,
                                 'Meteo')
        self.buffer = deque([], self.stack)
        self.shape = self.renderer.observe().shape + (self.stack,)

    def reset(self):
        if self.stack > 1:
            frame = self.renderer.observe()
            for i in range(self.stack):
                self.buffer.extend([frame])
            return np.dstack(list(self.buffer))
        return np.expand_dims(self.renderer.observe(), axis=-1)

    def observe(self):
        if self.show:
            self.renderer.show()
        if self.stack > 1:
            self.buffer.extend([self.renderer.observe()])
            return np.dstack(list(self.buffer))
        return np.expand_dims(self.renderer.observe(), axis=-1)


class Environment:
    def __init__(self, show=True):
        self.fps = 15
        self.view = View(self.fps, show)
        self.actions = ['a',  # move left
                        'd',  # move right
                        ' ']  # idle
        UI.register_key('a', watch=False)
        UI.register_key('d', watch=False)
        # UI.register_key('q', self.quit)
        self.reset()
        self.closest_meteor = 1
        self.done = False

    def toggle_show(self, show=None):
        if show is None:
            self.view.show = not self.view.show
        else:
            self.view.show = show

    def quit(self):
        self.view.renderer.close()
        self.done = True

    def meteor_close(self, d):
        from src.environment.game.objects import Object
        self.closest_meteor = min(self.closest_meteor, (Object.radius+d)/Object.radius)
        if self.closest_meteor <= 0:
            self.done = True

    def step(self, action=None):
        reward = 1
        self.closest_meteor = 1
        if action is None:
            action = self.actions[-1]
        if action in self.actions[:-1]:
            UI.press(action)
        self.game.update(UI.keys)
        if action in self.actions[:-1]:
            UI.release(action)
            reward -= 0.2
        reward -= 1-self.closest_meteor
        if self.done:
            reward = -1
        self.closest_meteor = 1
        return self.view.observe(), reward, self.done

    def state_shape(self):
        return self.view.shape

    def reset(self):
        self.done = False
        self.closest_meteor = 1
        self.game = Game(self.fps, self.view.aspect, self.view.renderer)
        self.game.add_player(objects.Rocket(self.actions, self.meteor_close))
        return self.view.reset()
