import lcapi
from . import globalvars
from .globalvars import get_global_device as device
from . import Buffer, Texture2D

class BindlessArray:
    def __init__(self, n_slots = 65536):
        self.handle = device().impl().create_bindless_array(n_slots)

    def emplace(self, idx, res):
        if type(res) is Buffer:
            device().impl().emplace_buffer_in_bindless_array(self.handle, idx, res.handle, 0)
        elif type(res) is Texture2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Texture2D must be float")
            sampler = lcapi.Sampler(lcapi.Sampler.Filter.POINT, lcapi.Sampler.Address.EDGE)
            device().impl().emplace_tex2d_in_bindless_array(self.handle, idx, res.handle, sampler)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def remove(self, res):
        if type(res) is Buffer:
            device().impl().remove_buffer_in_bindless_array(self.handle, res.handle)
        elif type(res) is Texture2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Texture2D must be float")
            device().impl().remove_tex2d_in_bindless_array(self.handle, res.handle)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def __contains__(self, res):
        return device().impl().is_resource_in_bindless_array(self.handle, res.handle)

    def update(self, stream = None):
        if stream is None:
            stream = globalvars.stream
        cmd = lcapi.BindlessArrayUpdateCommand.create(self.handle)
        stream.add(cmd)