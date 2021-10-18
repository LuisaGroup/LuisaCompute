from ctypes import CDLL
from os import environ

INSTALL_DIR = environ["LUISA_COMPUTE_INSTALL_DIR"]
dll = CDLL(f"{INSTALL_DIR}/libluisa-compute-api.dylib")
