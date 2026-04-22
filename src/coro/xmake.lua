-- NOTE: build coroutine/XIR integration primarily with xmake.
-- NOTE: keep coroutine regression coverage in src/xir/tests, especially xir_test_materialize_coro.
-- NOTE: coroutine runtime samples/tests live under src/coro/test and should be built via xmake targets here.
-- NOTE: add new coroutine tests under src/coro/test and xmake will pick them up automatically.
local lc_enable_gui = has_config("lc_enable_gui")

target("lc-coro")
set_basename("luisa-coro")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
set_pcxxheader("pch.h")
add_deps("lc-runtime", "lc-dsl", "lc-xir")
add_headerfiles("../../include/luisa/coro/**.h")
add_files("*.cpp", "schedulers/*.cpp")
add_defines("LUISA_CORO_EXPORT_DLL")
target_end()

local function coro_test(name, opts)
    opts = opts or {}
    if opts.gui and not lc_enable_gui then
        return
    end
    target("coro_test_" .. name)
    _config_project({
        project_kind = "binary"
    })
    add_deps("lc-backends-dummy", {
        inherit = false,
        links = false
    })
    set_pcxxheader("pch.h")
    add_files(path.join("test", name .. ".cpp"))
    add_includedirs(path.join(os.scriptdir(), "../tests/common"))
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-coro", "stb-image")
    if opts.gui then
        add_deps("lc-gui")
        add_defines("LUISA_ENABLE_GUI")
    end
    target_end()
end

local gui_tests = {
    path_tracing_persistent_threads = true,
    path_tracing_state_machine = true,
    path_tracing_wavefront = true
}

for _, test_file in ipairs(os.files(path.join(os.scriptdir(), "test", "*.cpp"))) do
    local test_name = path.basename(test_file)
    coro_test(test_name, {gui = gui_tests[test_name] == true})
end
