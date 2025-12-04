includes("common")
target("lc_backend_sdk")
do
    set_kind("phony")
    local libnames = {}
    if os.host() == "windows" and (has_config("lc_dx_backend") or has_config("lc_vk_backend")) then
        table.insert(libnames, 'dx_sdk')
    end
    if os.host() == "linux" and os.arch() == "x86_64" and has_config("lc_vk_backend") then
        table.insert(libnames, 'vk_sdk')
    end
    if #libnames > 0 then
        add_rules('lc_install_sdk', {
            libnames = libnames
        })
    end
end
target_end()

if has_config("lc_dx_backend") then
    includes("dx")
end
if has_config("lc_cuda_backend") then
    includes("cuda")
end
if has_config("lc_metal_backend") then
    includes("metal")
end
if has_config("lc_fallback_backend") then
    includes("fallback")
end
if has_config("lc_vk_backend") then
    includes("vk")
end
includes("validation")
if has_config("lc_toy_c_backend") then
    includes("toy_c")
end

rule("lc-backend-deps")
on_load(function(target)
    target:add("deps", "lc-validation-layer", {
        inherit = false,
        links = false
    })
    -- target:add("deps", "lc-backend-toy-c", {
    --     inherit = false
    -- })
    target:add("deps", "lc_backend_sdk");
    if has_config("lc_dx_backend") then
        target:add("deps", "lc-backend-dx", {
            inherit = false,
            links = false
        })
    end
    if has_config("lc_cuda_backend") then
        target:add("deps", "lc-backend-cuda", {
            inherit = false,
            links = false
        })
    end
    if has_config("lc_metal_backend") then
        target:add("deps", "lc-backend-metal", {
            inherit = false,
            links = false
        })
    end
    if has_config("lc_vk_backend") then
        target:add("deps", "lc-backend-vk", {
            inherit = false,
            links = false
        })
    end
    if has_config("lc_fallback_backend") then
        target:add("deps", "lc-backend-fallback", {
            inherit = false,
            links = false
        })
    end
end)
rule_end()

target("lc-backends-dummy")
set_kind("phony")
add_rules("lc-backend-deps")
target_end()
