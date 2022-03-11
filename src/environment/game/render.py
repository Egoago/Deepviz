from typing import Tuple

import cv2
import numpy as np


class Renderer:
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
        pass

    def draw(self, pos: np.ndarray, scale: float, path: str):
        pass

    def show(self):
        pass

    def observe(self):
        pass


class OpenCVRenderer(Renderer):
    imgs = {}

    def close(self):
        cv2.destroyAllWindows()

    def save_to(self, path):
        self.video = cv2.VideoWriter(path+'.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.fps*2, tuple(self.res))

    def __del__(self):
        if self.video is not None:
            self.video.release()

    def draw(self, pos: np.ndarray, scale: float, path: str):
        if path not in OpenCVRenderer.imgs:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            size = int(scale*self.res.shape[0])
            OpenCVRenderer.imgs[path] = cv2.resize(img, (size, size))
        img = OpenCVRenderer.imgs[path]
        r = img.shape[0]//2
        x1, y1, = pos[0]-r, pos[1]-r
        x2, y2 = x1+img.shape[1], y1+img.shape[0]

        mask = img[:, :, 3:] / 255
        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2] * (1 - mask) + img[:, :, :3] * mask

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
            scale = 2**self.downscale_factor
            showed = cv2.resize(gray_scale,
                                (int(gray_scale.shape[1] * scale),
                                 int(gray_scale.shape[0] * scale)),
                                interpolation=cv2.INTER_NEAREST)
            cv2.imshow('observed', showed)
            cv2.waitKey(1)
        return gray_scale
