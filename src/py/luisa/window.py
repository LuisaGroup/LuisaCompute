from typing import Callable
import dearpygui.dearpygui as dpg
import array

float4 = tuple[float, float, float, float]
int2 = tuple[int, int]

class Window:
    def __init__(self, name: str, initial_size: int2, resizable: bool = False, frame_rate: bool = False) -> None:
        self.resizable = resizable
        self.initial_size = initial_size
        self.texture = None
        self.frame_rate = None
        dpg.create_context()
        dpg.create_viewport(title=name,
                            width=initial_size[0],
                            height=initial_size[1],
                            resizable=resizable)
        if frame_rate:
            self.frame_rate = dpg.add_window(label="Frame rate", pos=(0, 0))
            dpg.add_text('0.0', tag="frame_rate_text", parent=self.frame_rate)
        dpg.add_viewport_drawlist(front=False, tag="viewport_draw")
        dpg.setup_dearpygui()
        dpg.set_viewport_vsync(False)
        dpg.show_viewport()

    def update_frame_rate(self, value: float):
        dpg.configure_item('frame_rate_text', default_value=str(value))

    def run(self, draw: Callable) -> None:
        while dpg.is_dearpygui_running():
            self.__begin_frame()
            draw()
            self.__end_frame()

    def set_background(self, pixels: array.ArrayType, size: int2) -> None:
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(size[0], size[1], pixels, format=dpg.mvFormat_Float_rgba, tag="background")

    def __begin_frame(self) -> None:
        pass

    def __end_frame(self) -> None:
        dpg.render_dearpygui_frame()
