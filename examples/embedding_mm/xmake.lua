-- EmbeddingGemma-2 multimodal embedding engine (luisa::embed_mm).

target("lc-example-embedding-mm")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "static"
    })
    add_files("embed/*.cpp", "embed/ops/*.cpp", "embed/models/*.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/src/ext/yyjson/src")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-yyjson")
    add_includedirs("$(projectdir)/src/ext/yyjson/src", {
        public = true
    })
    add_includedirs("$(projectdir)/examples/embedding_mm", {
        public = true
    })
target_end()

target("example_embed_mm_pack_model")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("tools/pack_model.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_text")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_text.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_test")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_test.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_vision")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_vision.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_audio")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_audio.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_video")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_video.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()

target("example_embed_mm_bench")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "binary"
    })
    add_files("example_embed_mm_bench.cpp")
    add_includedirs("$(projectdir)/examples", "$(projectdir)/examples/embedding_mm")
    add_deps("lc-example-embedding-mm")
target_end()
