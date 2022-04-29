import numpy as np
import luisa
from luisa.mathtypes import *

luisa.init()
# ============= test script ================


@luisa.kernel
def f():
    a = 1
    if dispatch_id().x < 5 and dispatch_id().y < 5:
        print("test12333", dispatch_id())


f(dispatch_size=(1024, 1024, 1))
luisa.synchronize()

accel = get_global_device().create_accel(lcapi.AccelUsageHint.FAST_TRACE)
v_buffer = luisa.Buffer(3, float3)
t_buffer = luisa.Buffer(3, int)
v_buffer.copy_from(np.array([0,0,0,0,0,1,2,0,0,2,1,0], dtype=np.float32))
t_buffer.copy_from(np.array([0,1,2], dtype=np.int32))
mesh_handle = get_global_device().impl().create_mesh(v_buffer, 0, 16, v_count, t_buffer, 0, t_count, lcapi.AccelUsageHint.FAST_TRACE)
globalvars.stream.add(lcapi.MeshBuildCommand.create(mesh_handle, lcapi.BuildRequest.PREFER_UPDATE, v_buffer, t_buffer))
accel.emplace_back(mesh_handle, float4x4.identity(), True)

globalvars.stream.add(accel.build_command())
globalvars.stream.synchronize()

# arr = np.ones(1024*1024*4, dtype=np.uint8)
# img.copy_to(arr)
# print(arr)
# im.fromarray(arr.reshape((1024,1024,4))).save('aaa.png')
# cv2.imwrite("a.hdr", arr.reshape((1024,1024,4)))

