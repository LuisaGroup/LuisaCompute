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
-- The TIRx bridge stays excluded unless lc_tile_tirx_bridge is enabled, in
-- which case the bundled apache/tvm submodule (src/ext/tvm) is built from
-- source by src/ext/xmake.lua and linked directly.
add_files("*.cpp", "bridge/xir/**.cpp")
-- The TIRx bridge translation units define same-named anonymous-namespace
-- helpers (e.g. ElementDomain) that collide when merged by the unity build,
-- so they are always compiled as standalone units.
add_files("bridge/tirx/**.cpp", {unity_ignored = true})
add_defines("LUISA_TILE_XIR_BRIDGE_EXPORT_DLL")

on_load(function(target)
      if has_config("lc_tile_tirx_bridge") then
          target:add("defines", "LUISA_TILE_TIRX_BRIDGE_EXPORT_DLL")
          -- The bundled TVM/tvm-ffi headers use throw, so this target needs
          -- C++ exceptions even though the project default disables them.
          target:set("exceptions", "cxx")
        -- The TVM targets carry the tvm/include, tvm-ffi/include and dlpack
        -- include directories in their public interface, so depending on them
        -- wires up both the headers and the linker inputs.
        target:add("deps", "tvm_compiler", "tvm_runtime", "tvm_ffi")
        -- tvm_ffi.dll / tvm_runtime.dll / tvm_compiler.dll live next to
        -- luisa-tile in the shared bin directory.
        if is_plat("macosx") then
            target:add("rpathdirs", "@loader_path")
        elseif is_plat("linux") then
            target:add("rpathdirs", "$ORIGIN")
        end
    else
        target:add("remove_files", "bridge/tirx/*.cpp")
    end
end)

on_config(function(target)
    if has_config("lc_tile_tirx_bridge") then
        local tvm_cmake = path.join(get_config("lc_ext_path"), "tvm", "CMakeLists.txt")
        if not os.exists(tvm_cmake) then
            raise("lc_tile_tirx_bridge requires the bundled TVM submodule. " ..
                  "Run `git submodule update --init src/ext/tvm` (with " ..
                  "`3rdparty/tvm-ffi` and its `3rdparty/dlpack`).")
        end
    end
end)
target_end()
