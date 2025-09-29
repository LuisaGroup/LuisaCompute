local script_dir = path.join(os.scriptdir(), "metal_builtin")
local dst_script_dir = os.scriptdir()
function src_dir()
    return script_dir
end

function dst_file()
    return path.join(dst_script_dir, "metal_builtin_embedded.cpp")
end

function meta_dir()
    return path.join(script_dir, "_meta.msi")
end

function file_list()
    return {'metal_builtin_kernels.metal', 'metal_device_lib.metal'}
end