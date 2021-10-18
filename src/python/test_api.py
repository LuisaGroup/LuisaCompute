import LuisaCompute as lc

if __name__ == "__main__":
    lc.set_log_level_verbose()
    ctx = lc.Context()
    device = lc.Device(ctx, "metal")
    buffer = device.create_buffer(lc.Mat2, 1024)
    t = lc.Type.struct([int, float], lc.Float4, alignment=8)
    print(t)
    print(t.members)
    print(t.is_array)
