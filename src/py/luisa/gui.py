import numpy as np
import lcapi
from . import globalvars
from .globalvars import stream, get_global_device

class GUI:
    def __init__(self, title: str, resolution, vsync = True, show_FPS = True):
        if not globalvars.stream_support_gui:
            raise RuntimeError("Stream no support GUI, use init(support_gui=True).")
        self.window = lcapi.PyWindow()
        self.window.reset(get_global_device(), globalvars.stream, title, resolution[0], resolution[1], vsync)
        self.resolution = resolution
        self.clock = lcapi.Clock()
        self.tic = False

    def show(self):
        self.window.present(globalvars.stream, self.tex.handle, self.resolution[0], self.resolution[1], 0, self.tex.storage)
        if self.tic:
            time = self.clock.toc()
        else:
            time = 0.0
        self.clock.tic()
        self.tic = True
        return time

    def running(self):
        return not self.window.should_close()

    def set_image(self, tex):
        self.tex = tex