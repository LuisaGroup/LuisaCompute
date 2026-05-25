-- NOTE: XIR coroutine materialization should be built and verified via xmake.
-- NOTE: Add XIR regression tests under src/xir/tests as test_*.cpp files.
target("lc-xir")
set_basename("luisa-xir")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime")
add_headerfiles("../../include/luisa/xir/**.h")
add_files("*.cpp", "instructions/**.cpp", "metadata/*.cpp", "translators/*.cpp", "passes/*.cpp", "passes/**.cpp")
add_defines("LUISA_XIR_EXPORT_DLL")
add_defines("LUISA_ENABLE_XIR=1", {
    public = true
})
on_load(function(target)
    if has_config("lc_yyjson_use_xrepo") then
        target:add("packages", "yyjson")
    else
        target:add("deps", "lc-yyjson")
    end
end)
target_end()

local function xir_test(name)
    target("xir_test_" .. name)
    _config_project({
        project_kind = "binary"
    })
    add_deps("lc-backends-dummy", {
        inherit = false,
        links = false
    })
    add_deps("lc-runtime", "lc-vstl", "lc-dsl", "lc-xir")
    add_files(path.join("tests", "test_" .. name .. ".cpp"))
    target_end()
end

xir_test("aggregate_field_bitmasks")
xir_test("mem2reg")
xir_test("debug_printer")
xir_test("dce")
xir_test("ray_query")
xir_test("materialize_coro")
xir_test("control_flow")
xir_test("control_flow_dump")
xir_test("xir2ast")
target("xir_test_materialize_coro")
add_deps("lc-ir", "lc-coro")
target_end()
target("xir_test_xir2ast")
add_deps("lc-ir")
target_end()
if has_config("lc_enable_gui") then
    xir_test("xir2ast_mpm")
    target("xir_test_xir2ast_mpm")
    add_deps("lc-ir", "lc-gui")
    add_defines("LUISA_ENABLE_GUI")
    target_end()
end
