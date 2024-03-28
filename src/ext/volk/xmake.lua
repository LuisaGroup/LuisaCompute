target("volk")
set_kind("headeronly")
add_includedirs(".", {
    public = true
})
add_defines("VK_NO_PROTOTYPES", {
    public = true
})
on_load(function(target)
    if is_plat("windows") then
        local sdk_dir = os.getenv("VK_SDK_PATH")
        if not sdk_dir then
            sdk_dir = os.getenv("VULKAN_SDK")
        end
        if not sdk_dir then
            utils.error("Vulkan not found.")
            target:set_enabled(false)
            return
        end
        target:add("includedirs", path.join(sdk_dir, "Include"), {
            public = true
        })
    end
end)
target_end()
