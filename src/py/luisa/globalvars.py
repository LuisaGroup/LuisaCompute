# global variables
context = None
device = None
stream = None

def get_global_device():
	if device == None:
		raise RuntimeError("device is None. Did you forget to call luisa.init()?")
	return device
