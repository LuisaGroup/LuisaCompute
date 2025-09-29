local script_dir = path.join(os.scriptdir(), "cuda_builtin")
local dst_script_dir = os.scriptdir()

function src_dir()
    return script_dir
end

function dst_file()
    return path.join(dst_script_dir, "cuda_builtin_embedded.cpp")
end

function meta_dir()
    return path.join(script_dir, "_meta.msi")
end

function file_list()
    return {'cuda_builtin_kernels.cu', 'cuda_device_coop.h', 'cuda_device_half.h', 'cuda_device_math.h', 'cuda_device_resource.h'}
end