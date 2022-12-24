from luisa import *
from luisa.builtin import *
init_headless()
log_level_info()
mip_level = 0
block_size = 0
Shared = None
img = Texture2DType(float, 4)
@func
def mip_gen(a0, a1, a2, a3, a4, a5):
    set_block_size(block_size, block_size, 1)
    block = Shared()
    thd = thread_id().xy
    grp = block_id().xy
    value = a0.read(dispatch_id().xy)
    actived_block = block_size
    for i in range(mip_level):
        next_actived_block = actived_block // 2
        if all(thd < int2(actived_block)):
            block[thd.x + thd.y * actived_block] = value
        sync_block()
        if all(thd < int2(next_actived_block)):
            value = \
                block[thd.x * 2 + 1 + (thd.y * 2) * actived_block] + \
                block[thd.x * 2 + (thd.y * 2) * actived_block] + \
                block[thd.x * 2 + 1 + (thd.y * 2 + 1) * actived_block] + \
                block[thd.x * 2 + (thd.y * 2 + 1) * actived_block]
            value *= 0.25
            idx = grp * int2(next_actived_block) + thd
            if i == 0:
                a1.write(idx, value)
            if mip_level >= 2:
                if i == 1:
                    a2.write(idx, value)
            if mip_level >= 3:
                if i == 2:
                    a3.write(idx, value)
            if mip_level >= 4:
                if i == 3:
                    a4.write(idx, value)
            if mip_level >= 5:
                if i == 4:
                    a5.write(idx, value)
        actived_block = next_actived_block
    
imgs = [img]
for i in range(1, 6):
    block_size = 1 << i
    mip_level = i
    Shared = SharedArrayType(block_size * block_size, float4)
    imgs.append(img)
    mip_gen.save(imgs, "shaders/_mip_gen" + str(i))

print("compile finished")