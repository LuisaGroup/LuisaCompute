target("lc-tile")
set_basename("luisa-tile")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
-- lc-runtime carries the XIR symbols used by the always-on tile/XIR bridge
-- (bridge/xir). The CMake build keeps that bridge as a separate static lib
-- linking luisa-compute-xir; xmake merges XIR into lc-runtime, so lc-tile
-- links the runtime DLL directly.
add_deps("lc-core", "lc-runtime")
add_defines("LUISA_TILE_EXPORT_DLL")
add_headerfiles("../../include/luisa/tile/**.h", "../../include/luisa/tile.h")
-- Include the whole tile source tree so the always-on XIR bridge and the
-- optional TVM TIRx bridge are compiled, mirroring the CMake source layout.
-- The TIRx bridge stays excluded unless lc_tile_tirx_bridge is enabled with
-- TVM include/library paths (same optionality as LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE).
add_files("**.cpp")
add_defines("LUISA_TILE_XIR_BRIDGE_EXPORT_DLL")

local function tirx_paths_configured()
    local names = {
        "lc_tvm_include_dir",
        "lc_tvm_ffi_include_dir",
        "lc_tvm_library_dir",
        "lc_tvm_ffi_library_dir"
    }
    for _, name in ipairs(names) do
        local value = get_config(name)
        if type(value) ~= "string" or value == "" then
            return false
        end
    end
    return true
end

on_load(function(target)
    if has_config("lc_tile_tirx_bridge") then
        target:add("defines", "LUISA_TILE_TIRX_BRIDGE_EXPORT_DLL")
        if tirx_paths_configured() then
            local tvm_include = get_config("lc_tvm_include_dir")
            local tvm_ffi_include = get_config("lc_tvm_ffi_include_dir")
            local tvm_library_dir = get_config("lc_tvm_library_dir")
            local tvm_ffi_library_dir = get_config("lc_tvm_ffi_library_dir")
            target:add("includedirs", tvm_include, tvm_ffi_include)
            local dlpack_include = path.join(tvm_ffi_include, "../3rdparty/dlpack/include")
            if os.exists(dlpack_include) then
                target:add("includedirs", path.normalize(dlpack_include))
            end
            target:add("linkdirs", tvm_library_dir, tvm_ffi_library_dir)
            target:add("links", "tvm_compiler", "tvm_runtime", "tvm_ffi")
        end
    else
        target:add("remove_files", "bridge/tirx/*.cpp")
    end
end)

on_config(function(target)
    if has_config("lc_tile_tirx_bridge") and not tirx_paths_configured() then
        raise("lc_tile_tirx_bridge requires --lc_tvm_include_dir, " ..
              "--lc_tvm_ffi_include_dir, --lc_tvm_library_dir and " ..
              "--lc_tvm_ffi_library_dir (mirrors the CMake TVM options).")
    end
end)
target_end()
