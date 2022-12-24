import lcapi
from .types import to_lctype, from_lctype

# Note: vector & matrix types are directly imported from lcapi

# class Vector:
    # def __init__(self, data, dtype = float):
    #     if not dtype in {int, float, bool}:
    #         raise Exception('invalid vector dtype')
    #     self.dtype = dtype
    #     self.data = np.array(data, dtype={int:np.int32, float:np.float32, bool:bool}[dtype])
    #     if len(self.data.shape) != 1:
    #         raise Exception('invalid vector shape')
    #     if not self.data.size in {2,3,4}:
    #         raise Exception('vector len must be 2/3/4')
    #     self.size = self.data.size

# @staticmethod
def is_swizzle_name(sw):
    if len(sw) > 4:
        return False
    for ch in sw:
        if not ch in {'x','y','z','w'}:
            return False
    return True

# @staticmethod
def get_swizzle_code(sw, maxlen):
    code = 0
    codemap = {
        'x': 0,
        'y': 1,
        'z': 2,
        'w': 3,
    }
    for idx,ch in enumerate(sw):
        c = codemap[ch]
        if c >= maxlen:
            raise Exception('swizzle index exceeding length of vector')
        code |= c << (idx * 4)
    return code

def get_swizzle_resulttype(dtype, len):
    lctype = to_lctype(dtype)
    if len == 1:
        return from_lctype(lctype.element())
    else:
        return from_lctype(lcapi.Type.from_(f'vector<{lctype.element().description()},{len}>'))
