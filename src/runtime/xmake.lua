target("lc-runtime")
set_basename("luisa-runtime")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-core", "lc-vstl")
-- json2ast/xir2json include <yyjson.h>; the bundled lc-yyjson dep must be
-- registered at target scope: an add_deps issued from on_load is too late for
-- link resolution. The xrepo package variant is handled in on_load below.
if not has_config("lc_yyjson_use_xrepo") then
    add_deps("lc-yyjson")
end
  lc_set_pcxxheader("lc_runtime_pch.h")
  add_defines("LUISA_RUNTIME_EXPORT_DLL", "LUISA_AST_EXPORT_DLL", "LUISA_XIR_EXPORT_DLL")
  add_headerfiles("../../include/luisa/runtime/**.h", "../../include/luisa/ast/**.h")
on_load(function(target)
    if has_config('lc_safe_mode') then
        target:add('defines', 'LUISA_ENABLE_SAFE_MODE', {public = true})
    end
    if has_config("lc_enable_xir") then
        target:add("defines", "LUISA_ENABLE_XIR", {public = true})
          if has_config("lc_yyjson_use_xrepo") then
              target:add("packages", "yyjson")
          else
              target:add("deps", "lc-yyjson")
              target:add("includedirs", path.absolute("../ext/yyjson/src", os.scriptdir()), {public = true})
          end
    end
    if target:is_plat("windows") then
        target:add("cxxflags", "/bigobj", {tools = "cl"})
    end
    target:add("files", path.absolute("../ast/*.cpp", os.scriptdir()), path.join(os.scriptdir(), "**.cpp"))
    if has_config("lc_enable_xir") then
        local ir_path = path.absolute("../xir", os.scriptdir())
        target:add("files", path.join(ir_path, "*.cpp"), path.join(ir_path, "instructions/*.cpp"),
            path.join(ir_path, "metadata/*.cpp"), path.join(ir_path, "translators/*.cpp"),
            path.join(ir_path, "passes/*.cpp"))
    end
end)
target_end()
