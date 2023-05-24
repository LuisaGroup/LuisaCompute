from luisa import *
# use headless mode, no runtime available
log_level_error()
init_headless()

@func
def out_kernel(tex):
    set_block_size(16, 16, 1)
    index = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(index) + 0.5) / float2(size)
    tex.write(index, float4(uv, 0.5, 1.))


# save this kernel as "out_kernel.bytes"
cpp_header = out_kernel.save(
    (Texture2DType(float, 4), ),
    "out_kernel.bytes",
    # compile in independent thread
    async_build=True,
    # print c++ header(maybe useful for c++ runtime)
    print_cpp_header=True
)
print("Compiled success, C++ header:\n")
print(cpp_header)
