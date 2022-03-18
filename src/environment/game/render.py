from typing import Tuple
import unittest
from os.path import exists
import cv2
import numpy as np


class Renderer:
    imgs = {}
    sprite_dir = "../media/sprites/"

    def __init__(self, res: Tuple[int, int], fps, title=''):
        self.title = title
        self.res = np.asarray(res)
        self.fps = fps
        self.frame = self.clear()
        self.downscale_factor = 2
        self.video = None

    def __in_pixels__(self, coords: np.ndarray) -> Tuple[int, int]:
        assert len(coords) == 2
        return round(coords[0]*self.res[0]), round(self.res[1]-coords[1]*self.res[0])

    def clear(self) -> np.ndarray:
        self.frame = np.zeros(([self.res[1], self.res[0], 3]), np.uint8)
        return self.frame

    def close(self):
        cv2.destroyAllWindows()

    def draw(self, pos: np.ndarray, scale: float, path: str):
        if path is None:
            return
        if path not in Renderer.imgs:
            if not exists(Renderer.sprite_dir+path):
                raise Exception(f"Image {path} not found")
            img = cv2.imread(Renderer.sprite_dir+path, cv2.IMREAD_UNCHANGED)
            size = int(scale * self.res[0])
            Renderer.imgs[path] = cv2.resize(img, (size, size))
        img = Renderer.imgs[path]
        center = (pos*self.res[0]).astype(int)
        center[1] = self.res[1]-center[1]
        x1 = min(max(int(center[0] - img.shape[0]//2), 0), self.res[0])
        y1 = min(max(int(center[1] - img.shape[1]//2), 0), self.res[1])
        x2 = max(min(int(center[0] + img.shape[0]//2), self.res[0]), 0)
        y2 = max(min(int(center[1] + img.shape[1]//2), self.res[1]), 0)

        sprite_x1 = img.shape[0]//2+x1-center[0]
        sprite_x2 = img.shape[0]//2+x2-center[0]
        sprite_y1 = img.shape[1]//2+y1-center[1]
        sprite_y2 = img.shape[1]//2+y2-center[1]

        mask = img[sprite_y1:sprite_y2, sprite_x1:sprite_x2, 3:] / 255
        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2] * (1 - mask) + img[sprite_y1:sprite_y2, sprite_x1:sprite_x2, :3] * mask

    def show(self):
        if self.video is not None:
            self.video.write(self.frame)
        cv2.imshow(self.title, self.frame)
        cv2.waitKey(1)

    def observe(self):
        small = self.frame.copy()
        for i in range(self.downscale_factor):
            small = cv2.pyrDown(small)
        gray_scale = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if False:
            scale = 2 ** self.downscale_factor
            showed = cv2.resize(gray_scale,
                                (int(gray_scale.shape[1] * scale),
                                 int(gray_scale.shape[0] * scale)),
                                interpolation=cv2.INTER_NEAREST)
            cv2.imshow('observed', showed)
            cv2.waitKey(1)
        return gray_scale

    def render_to(self, path):
        self.video = cv2.VideoWriter(path+'.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.fps*2, tuple(self.res))

    def __del__(self):
        if self.video is not None:
            self.video.release()


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer((800, 800), 30, "test")

    def test_render_img(self):
        path = "rocket.png"
        self.renderer.draw(np.array([0.5, -0.5]), 0.2, path)
        self.renderer.show()
        cv2.waitKey(2000)
