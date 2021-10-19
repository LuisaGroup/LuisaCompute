import numpy as np
import LuisaCompute as lc

if __name__ == "__main__":
    lc.set_log_level_verbose()
    device = lc.Device("metal", 1)
    buffer = device.create_buffer(lc.float4, 4096)
    view = buffer[1024:2048]
    in_data = np.array(np.random.rand(1024, 4), dtype=np.float32)
    out_data = np.empty([1024 * 4], dtype=np.float32)
    with device.default_stream:
        view.upload(in_data)
        view.download(out_data)
    with device.default_stream as s:
        s.synchronize()
    print(in_data)
    print(out_data)
    assert (in_data.reshape([-1]) == out_data).all()
