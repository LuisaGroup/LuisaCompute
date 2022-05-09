import dearpygui.dearpygui as dpg
import numpy as np
from .framerate import FrameRate

class GUI:
    def __init__(self, title, resolution, resizable = False, show_FPS = True):
        dpg.create_context()
        dpg.create_viewport(title=title,
                            width=resolution[0],
                            height=resolution[1],
                            resizable=False)
        self.frame_rate_window = None
        self.frame_rate = None
        if show_FPS:
            self.frame_rate_window = dpg.add_window(label="Frame rate", pos=(0, 0))
            dpg.add_text('N/A', tag="frame_rate_text", parent=self.frame_rate_window)
        dpg.add_viewport_drawlist(front=False, tag="viewport_draw")
        dpg.setup_dearpygui()
        dpg.set_viewport_vsync(False)
        dpg.show_viewport()

        self.texture_array = np.zeros((*resolution,4), dtype=np.float32)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(resolution[0], resolution[1], self.texture_array, format=dpg.mvFormat_Float_rgba, tag="background")

        dpg.draw_image("background", (0, 0), resolution, parent="viewport_draw")

    def show(self, frames_in_flight=1):
        dpg.render_dearpygui_frame()
        if self.frame_rate == None:
            self.frame_rate = FrameRate(10)
        self.frame_rate.record(frames_in_flight)
        if hasattr(self, 'frame_rate_window'):
            dpg.configure_item('frame_rate_text', default_value=str(self.frame_rate.report()))

    def running(self):
        return dpg.is_dearpygui_running()

    def set_image(self, tex):
        tex.copy_to(self.texture_array, sync=True)

