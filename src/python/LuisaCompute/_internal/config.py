from os import environ
from ctypes import CDLL
from platform import system

INSTALL_DIR = environ["LUISA_COMPUTE_INSTALL_DIR"]
SYSTEM_NAME = system()
if SYSTEM_NAME == "Darwin" or SYSTEM_NAME == "Linux":
    dll = CDLL(f"{INSTALL_DIR}/libluisa-compute-api.so")
elif SYSTEM_NAME == "Windows":
    dll = CDLL(f"{INSTALL_DIR}/luisa-compute-api.dll")
else:
    raise EnvironmentError(f"Unsupported system: {SYSTEM_NAME}")
