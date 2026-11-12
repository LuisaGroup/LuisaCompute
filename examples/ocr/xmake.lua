-- OCR engine (luisa::ocr) -- Phase 2: portable tensor runtime and
-- correctness-first operators on top of the Phase 1 host artifacts.
-- No model is validated yet; see MODELS.md.
-- Host-only engine TUs must not include DSL device headers; the runtime/ops
-- device TUs are the only place DSL factories live (device-free factories).

target("lc-example-ocr")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "static"
    })
    add_files("ocr/*.cpp", "ocr/output/*.cpp", "ocr/runtime/*.cpp", "ocr/ops/*.cpp", "ocr/models/*.cpp")
    add_includedirs("$(projectdir)/examples")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    -- yyjson: follow the repository pattern (bundled lc-yyjson by default,
    -- xrepo package when lc_yyjson_use_xrepo is on).
    if has_config("lc_yyjson_use_xrepo") then
        add_packages("yyjson")
    else
        add_deps("lc-yyjson")
        add_includedirs("$(projectdir)/src/ext/yyjson/src", {
            public = true
        })
    end
    add_includedirs("$(projectdir)/examples/ocr", {
        public = true
    })
target_end()

target("example_ocr")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_ocr.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/ocr")
    add_deps("lc-example-ocr")
target_end()

-- example_ocr_bench is reserved for the performance/budget harness phase
-- (plan Phase 11); its source does not exist yet and is intentionally not
-- referenced here.
