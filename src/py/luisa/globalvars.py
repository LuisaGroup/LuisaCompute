# global variables
from .dylibs import lcapi
current_context = None
device = None
class Vars:
    def __init__(self) -> None:
        self.context = None
        self.stream = None
        self.stream_support_gui = False
    def __del__(self):
        if lcapi:
            lcapi.delete_all_swapchain()
        global device
        if self.stream:
            del self.stream
        if device:
            del device
            device = None
        if self.context:
            del self.context

vars = None
def get_global_device():
    global device
    return device
