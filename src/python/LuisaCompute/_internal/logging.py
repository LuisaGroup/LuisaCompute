from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t
from .config import dll


dll.luisa_compute_set_log_level_verbose.restype = None
dll.luisa_compute_set_log_level_verbose.argtypes = []


def set_log_level_verbose():
    dll.luisa_compute_set_log_level_verbose()


dll.luisa_compute_set_log_level_info.restype = None
dll.luisa_compute_set_log_level_info.argtypes = []


def set_log_level_info():
    dll.luisa_compute_set_log_level_info()


dll.luisa_compute_set_log_level_warning.restype = None
dll.luisa_compute_set_log_level_warning.argtypes = []


def set_log_level_warning():
    dll.luisa_compute_set_log_level_warning()


dll.luisa_compute_set_log_level_error.restype = None
dll.luisa_compute_set_log_level_error.argtypes = []


def set_log_level_error():
    dll.luisa_compute_set_log_level_error()


dll.luisa_compute_log_verbose.restype = None
dll.luisa_compute_log_verbose.argtypes = [c_char_p]


def log_verbose(msg):
    dll.luisa_compute_log_verbose(msg.encode())


dll.luisa_compute_log_info.restype = None
dll.luisa_compute_log_info.argtypes = [c_char_p]


def log_info(msg):
    dll.luisa_compute_log_info(msg.encode())


dll.luisa_compute_log_warning.restype = None
dll.luisa_compute_log_warning.argtypes = [c_char_p]


def log_warning(msg):
    dll.luisa_compute_log_warning(msg.encode())


dll.luisa_compute_log_error.restype = None
dll.luisa_compute_log_error.argtypes = [c_char_p]


def log_error(msg):
    dll.luisa_compute_log_error(msg.encode())
