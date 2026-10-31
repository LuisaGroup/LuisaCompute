-- EmbeddingGemma-2 multimodal embedding engine (luisa::mm_embed).

target("lc-example-mm-embedding")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "static"
    })
    add_files("embed/*.cpp", "embed/kernels/*.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/src/ext/yyjson/src")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-yyjson", "stb-image")
    add_includedirs("$(projectdir)/src/ext/yyjson/src", {
        public = true
    })
    add_includedirs("$(projectdir)/examples/mm_embedding", {
        public = true
    })
target_end()

target("example_mm_embed_pack_model")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("tools/pack_model.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_dump_golden")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("tools/dump_golden.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_text")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_mm_embed_text.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_test")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_mm_embed_test.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_vision")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_mm_embed_vision.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_audio")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_mm_embed_audio.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()

target("example_mm_embed_bench")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_mm_embed_bench.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/mm_embedding")
    add_deps("lc-example-mm-embedding")
target_end()


