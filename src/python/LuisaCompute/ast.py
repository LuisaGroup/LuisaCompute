from ._internal.language import *
from .type import *
import numpy as np


class Constant:
    def __init__(self, data):
        def view():
            array = np.array(data)
            scalar = Type.of(array.dtype)
            shape = array.shape
            dim = len(shape)
            if dim == 1:
                t = scalar
            elif dim == 2:
                assert 2 <= shape[1] <= 4
                t = Type.vector(scalar, dim)
            elif dim == 3:
                assert 2 <= shape[1] == shape[2] <= 4 and scalar == Type.float
                t = Type.matrix(shape[1])
            else:
                raise ValueError(
                    f"Invalid dimension for constant data: "
                    f"{dim} (shape = {shape})")
            return t, array

        self._elem, self._array = view()
        self._as_parameter_ = ast_create_constant_data(
            self.element,
            np.ascontiguousarray(self.array).ctypes.data,
            len(self.array))

    def __del__(self):
        ast_destroy_constant_data(self)

    @property
    def element(self):
        return self._elem

    @property
    def array(self):
        return self._array
