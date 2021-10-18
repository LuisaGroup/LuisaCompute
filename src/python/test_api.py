import LuisaCompute as lc
import numpy as np

if __name__ == "__main__":
    lc.set_log_level_verbose()
    ctx = lc.Context()
    device = ctx.create_device("metal")
    buffer = device.create_buffer(lc.float4, 1024)
    in_data = np.array(np.random.rand(1024, 4), dtype=np.float32)
    out_data = np.empty([1024 * 4], dtype=np.float32)
    stream = device.create_stream()
    with stream:
        buffer.upload(in_data)
        lc.commit()
        buffer.download(out_data)
        lc.synchronize()
    print(in_data)
    print(out_data)
