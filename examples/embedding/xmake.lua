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

-- C-FFI front-end for the retrieval steps: a shared library that exports
-- initialize / load-dir / query / update / unload / encode as documented C
-- functions (api/example_embed_retrieval_lib.h, api/README.md), so a host in
-- any language can drive the retriever through its own FFI instead of the
-- interactive console.
target("example_embed_retrieval_lib")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "shared"
    })
    add_files("api/*.cpp")
    add_headerfiles("api/*.h")
    add_defines("EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding")
    add_includedirs("$(projectdir)/examples/embedding/api", {
        public = true
    })
    add_deps("lc-example-embedding", "lc-runtime", "lc-dsl", "lc-vstl")
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
