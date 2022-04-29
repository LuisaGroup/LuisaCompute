import luisa
import numpy as np
from luisa.builtin import builtin_func
from luisa.mathtypes import *
from PIL import Image as im

luisa.init()
# ============= test script ================


@luisa.kernel
def f():
    a = 1
    if dispatch_id().x < 5 and dispatch_id().y < 5:
        print("test12333", dispatch_id())



f(dispatch_size=(1024, 1024, 1))
luisa.synchronize()

# arr = np.ones(1024*1024*4, dtype=np.uint8)
# img.copy_to(arr)
# print(arr)
# im.fromarray(arr.reshape((1024,1024,4))).save('aaa.png')
# cv2.imwrite("a.hdr", arr.reshape((1024,1024,4)))

