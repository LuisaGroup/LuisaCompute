if get_config("_lc_vk_sdk_dir") then
    target("volk")
    _config_project({
        project_kind = "static"
    })
    add_includedirs(".", {
        public = true
    })
    add_defines("VK_NO_PROTOTYPES", {
        public = true
    })
    add_files("volk.c")
    on_load(function(target)
        if is_plat("windows") then
            local sdk_dir = get_config("_lc_vk_sdk_dir")
            target:add("includedirs", path.join(sdk_dir, "Include"), {
                public = true
            })
        end
    end)
    target_end()
end
