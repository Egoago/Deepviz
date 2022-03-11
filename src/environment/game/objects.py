import numpy as np
import logging


class Object:
    radius = 0.07

    def __init__(self, icon_path=None, position=None, on_collision=None):
        self.icon_path = icon_path
        self.on_collision = on_collision
        self.position = np.array(position, float) if position else None

    def collides(self, other: "Object") -> bool:
        if any(x is None for x in [self.position,
                                   other.position]):
            return False
        d = self.position - other.position
        d = np.linalg.norm(d)
        if d < 2 * Object.radius:
            if self.on_collision is not None:
                self.on_collision()
            if other.on_collision is not None:
                other.on_collision()
            logging.debug(f"Collided {self} with {other}")
            return True
        return False

    def update(self, dt, t, keys) -> bool:
        return True


class Meteor(Object):
    speed = 1

    def __init__(self, position=None):
        super().__init__(icon_path="media/meteor.png",
                         position=position)
        self.velocity = np.array([0, Meteor.speed], float)

    def update(self, dt, t, keys) -> bool:
        self.position += dt * self.velocity
        return self.position[1] > 0


class Rocket(Object):
    def __init__(self, keys, on_collision):
        super().__init__(icon_path="media/rocket.png",
                         position=[0.5, Object.radius],
                         on_collision=on_collision)
        self.keys = keys
        self.speed = 0.3

    def update(self, dt, t, keys) -> bool:
        if keys[self.keys[0]]:
            self.position[0] -= self.speed * dt
        elif keys[self.keys[1]]:
            self.position[0] += self.speed * dt
        self.position[0] = min(1-Object.radius, self.position[0])
        self.position[0] = max(Object.radius, self.position[0])
        return True
