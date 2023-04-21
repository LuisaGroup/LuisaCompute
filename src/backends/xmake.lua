includes("common")
if get_config("dx_backend") then
    includes("dx")
end
if get_config("cuda_backend") then
    includes("cuda")
end
if get_config("metal_backend") then
    includes("metal")
end
if get_config("cpu_backend") then
    includes("cpu")
end
if LCRemoteBackend then
    includes("remote")
end
if get_config("vk_backend") then
    includes("vk")
end
includes("validation")
target("lc-backends-dummy")
set_kind("phony")
add_deps("lc-validation-layer", { inherit = false })
if get_config("dx_backend") then
    add_deps("lc-backend-dx", { inherit = false })
end
if get_config("cuda_backend") then
    add_deps("lc-backend-cuda", { inherit = false })
end
if get_config("metal_backend") then
    add_deps("lc-backend-metal", { inherit = false })
end
if get_config("vk_backend") then
    add_deps("lc-backend-vk", { inherit = false })
end
if get_config("vk_backend") or get_config("dx_backend") then
    add_deps("lc-hlsl-builtin", { inherit = false })
end
if get_config("cpu_backend") then
    add_deps("lc-backend-cpu", { inherit = false })
end
target_end()