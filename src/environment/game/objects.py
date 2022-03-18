import numpy as np
import logging


class Object:
    radius = 0.07

    def __init__(self, icon_path=None, position=None, on_collision=None, collision_radius=None):
        self.icon_path = icon_path
        self.collision_radius = collision_radius
        self.alive = True
        if self.collision_radius is None:
            self.collision_radius = Object.radius
        self.on_collision = on_collision
        self.position = np.array(position, float) if position else None

    def collides(self, other: "Object"):
        if any(x is None for x in [self.position,
                                   other.position]):
            return
        d = self.position - other.position
        d = np.linalg.norm(d)
        k = d
        d -= self.collision_radius + other.collision_radius
        if d < 0:
            if self.on_collision is not None:
                self.on_collision(d)
            if other.on_collision is not None:
                other.on_collision(d)
            logging.debug(f"Collided {self} with {other}")

    def update(self, dt, t, keys) -> bool:
        return self.alive


class Scheduler(Object):
    def __init__(self, delay, callback, repeat=-1):
        super().__init__(None)
        self.repeat = repeat
        self.delay = delay
        self.next_call = delay
        self.callback = callback

    def update(self, dt, t, keys) -> bool:
        if self.next_call <= t:
            self.callback()
            self.next_call += self.delay
            if self.next_call < t:
                self.next_call = t + self.delay
            if -1 < self.repeat:
                if self.repeat < 1:
                    return False
                else:
                    self.repeat -= 1
        return super().update(dt, t, keys)


class Meteor(Object):
    speed = 1

    def __init__(self, position=None):
        super().__init__(icon_path="meteor.png",
                         position=position)
        self.velocity = np.array([0, -Meteor.speed], float)

    def update(self, dt, t, keys) -> bool:
        self.position += dt * self.velocity
        if self.position[1] < 0:
            self.alive = False
            return self.alive
        return super().update(dt, t, keys)


class Rocket(Object):
    def __init__(self, keys, on_collision):
        super().__init__(icon_path="rocket.png",
                         position=[0.5, Object.radius],
                         on_collision=on_collision,
                         collision_radius=Object.radius*2)
        self.keys = keys
        self.speed = 0.3

    def update(self, dt, t, keys) -> bool:
        if keys[self.keys[0]]:
            self.position[0] -= self.speed * dt
        elif keys[self.keys[1]]:
            self.position[0] += self.speed * dt
        self.position[0] = min(1-Object.radius, self.position[0])
        self.position[0] = max(Object.radius, self.position[0])
        return super().update(dt, t, keys)
