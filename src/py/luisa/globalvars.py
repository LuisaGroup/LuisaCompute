# global variables
context = None
device = None
stream = None
current_context = None

# NOTE: DO NOT import these variables!
#       Import globalvars and use globalvars.stream instead.

def get_global_device():
    if device == None:
        raise RuntimeError("device is None. Did you forget to call luisa.init()?")
    return device
