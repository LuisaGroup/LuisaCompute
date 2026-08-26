import os


def find_entry_points(hlsl_path: str):
    with open(hlsl_path, 'r') as f:
        lines = f.readlines()

    entry_points = []
    for (i, line) in enumerate(lines):
        if line.startswith("[numthreads("):
            entry_points.append(lines[i + 1].split()[1].split("(")[0])
    return entry_points


def cross_compile(hlsl_path: str, entry_point: str):
    from os import system
    spv_name = f"{hlsl_path[:-len('.hlsl')]}_{entry_point}"
    # system(f"dxc -O3 -spirv -T cs_6_5 {hlsl_path} -HV 2016 -E {entry_point} -Fo {spv_name}.spv -no-warnings")
    # system(f"spirv-cross {spv_name}.spv --msl --msl-argument-buffers --msl-argument-buffer-tier 1 --msl-version 30000 > {spv_name}.metal")
    with open(f"{spv_name}.metal", "r") as f:
        src = f.read()
    # patch the unusable source
    src = src \
        .replace("struct type_cbCS", "struct alignas(16) type_cbCS") \
        .replace("constant type_cbCS* cbCS [[id(0)]]", "type_cbCS cbCS") \
        .replace("    texture2d<float> g_Input [[id(1)]];\n"
                 "    device type_RWStructuredBuffer_v4uint* g_OutBuff [[id(2)]];",
                 "    texture2d<float> g_Input [[id(1)]];\n"
                 "    ulong _;\n"
                 "    device type_RWStructuredBuffer_v4uint* g_OutBuff [[id(2)]];") \
        .replace(" [[id(1)]]", "") \
        .replace(" [[id(2)]]", "") \
        .replace(" [[id(3)]]", "") \
        .replace("(*spvDescriptorSet0.cbCS)", "spvDescriptorSet0.cbCS")
    with open(f"{spv_name}.patched.metal", "w") as f:
        f.write(src)
    return src


# 16-char per line
def string_to_hex_array(s: str, indent=4):
    hex_array = ""
    for i in range(0, len(s), 16):
        hex_array += " " * indent + ", ".join(f"0x{ord(c):02x}" for c in s[i:i + 16]) + ",\n"
    return hex_array


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    mtl_sources = []
    for hlsl in ["BC7Encode.hlsl", "BC6HEncode.hlsl"]:
        hlsl_path = os.path.join(base_dir, hlsl)
        entry_points = find_entry_points(hlsl_path)
        for entry_point in entry_points:
            mtl_src = cross_compile(hlsl_path, entry_point)
            mtl_sources.append((f"{hlsl[:-len('.hlsl')]}_{entry_point}", mtl_src + "\0"))
    # embed the shaders into C hex arrays in "../metal_tex_compress.inl.h"
    with open(os.path.join(base_dir, "../metal_tex_compress.inl.h"), "w") as f:
        for name, source in mtl_sources:
            f.write(f"static const char metal_tex_compress_{name}[] = {{\n{string_to_hex_array(source)}}};\n\n")
