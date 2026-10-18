-- bge-m3 embedding engine (embedding_model_plan.md).
-- Engine static lib: sources drop in under embed/ per milestone; the wildcard
-- keeps this file stable while subagents own only their sources.
-- Test binaries (test_embed_*) are registered in src/tests/xmake.lua; the
-- example_embed_text demo lands with milestone M6.

target("lc-example-embedding")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "static"
    })
    add_files("embed/*.cpp", "embed/kernels/*.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/src/ext/yyjson/src")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-yyjson")
    add_includedirs("$(projectdir)/src/ext/yyjson/src", {
        public = true
    })
    add_includedirs("$(projectdir)/examples/embedding", {
        public = true
    })
target_end()

-- offline weight packer: ONNX + .pt checkpoints -> model.bgm (host-only tool)
target("example_embed_pack_model")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("tools/pack_model.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding")
target_end()

-- Interactive incremental retrieval REPL and newline-delimited JSON-RPC server.
target("example_embed_retrieval")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_retrieval.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding")
    add_deps("lc-example-embedding")
    if is_plat("windows") then
        add_syslinks("ws2_32")
    end
target_end()

-- dev tool: dump golden intermediates from a packed model (M0 counterpart, §9.1 D4)
target("example_embed_dump_golden")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("tools/dump_golden.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding")
    add_deps("lc-example-embedding")
target_end()

-- M6 demo: encode sentences from the command line (plan §6)
target("example_embed_text")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_text.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding")
    add_deps("lc-example-embedding")
target_end()
