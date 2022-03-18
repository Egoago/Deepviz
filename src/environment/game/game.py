import random
from typing import List

from src.environment.game import objects
from src.environment.game.render import Renderer


class Game:
    def __init__(self, fps, aspect, renderer: Renderer):
        self.renderer = renderer

        # Timing
        self.t = 0
        self.target_dt = 1/fps

        # Scene
        self.height = 1/aspect
        self.objects: List[objects.Object] = []
        #self.objects.append(Scheduler(2, FallingObject.accelerate, 9))
        self.spawner = objects.Scheduler(0.5, self.spawn, -1)
        self.objects.append(self.spawner)
        self.__draw__()

    def add_player(self, player):
        self.objects.append(player)

    def spawn(self):
        new_pos = [random.random(), self.height+0.1]
        self.objects.append(objects.Meteor(new_pos))

    def update(self, keys):
        self.t += self.target_dt
        self.objects = [obj for obj in self.objects if obj.update(self.target_dt, self.t, keys)]
        for i, first in enumerate(self.objects[:-1]):
            for second in self.objects[i+1:]:
                first.collides(second)
        self.__draw__()

    def __draw__(self):
        self.renderer.clear()
        for obj in self.objects:
            self.renderer.draw(obj.position, obj.radius*2, obj.icon_path)
