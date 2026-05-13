#include "file_builder.h"
#include "bm25_scorer.h"
#include "string_similarity.h"
#include "porter_stemmer.h"
#include "query_performance_predictor.h"
#include "mmr_rerank.h"
#include "levenshtein_automaton.h"
#include <luisa/vstl/md5.h>
#include <luisa/core/logging.h>
#include <luisa/core/fiber.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace tokenize {

static constexpr std::array<uint8_t, 4> CACHE_MAGIC = {'F', 'B', 'C', 'H'};
static constexpr uint8_t CACHE_VERSION = 1;

FileBuilder::FileBuilder(luisa::vector<luisa::filesystem::path> paths,
                         luisa::filesystem::path output_path,
                         luisa::filesystem::path cache_path,
                         luisa::filesystem::path index_path,
                         int n,
                         double k1,
                         double b)
    : _paths(std::move(paths)),
      _output_path(std::move(output_path)),
      _cache_path(std::move(cache_path)),
      _index_path(std::move(index_path)),
      _n(n), _k1(k1), _b(b) {
    if (_cache_path.empty()) {
        _cache_path = _output_path;
        _cache_path += ".cache";
    }
    if (_index_path.empty()) {
        _index_path = _output_path;
        _index_path += ".index";
    }
    _tokenizer = NgramTokenizer(_n);
    if (_cache_valid() && luisa::filesystem::exists(_index_path)) {
        if (_load_cache()) {
            return;
        }
    }
    _scan_and_build();
    _save_cache();
}

static bool _is_valid_utf8(const char *data, size_t len) {
    size_t i = 0;
    while (i < len) {
        uint8_t c = static_cast<uint8_t>(data[i]);
        if (c == 0) return false;
        if (c < 0x80) {
            ++i;
        } else if (c < 0xC2) {
            return false;
        } else if (c < 0xE0) {
            if (i + 1 >= len) return false;
            uint8_t c1 = static_cast<uint8_t>(data[i + 1]);
            if ((c1 & 0xC0) != 0x80) return false;
            i += 2;
        } else if (c < 0xF0) {
            if (i + 2 >= len) return false;
            uint8_t c1 = static_cast<uint8_t>(data[i + 1]);
            uint8_t c2 = static_cast<uint8_t>(data[i + 2]);
            if ((c1 & 0xC0) != 0x80) return false;
            if ((c2 & 0xC0) != 0x80) return false;
            if (c == 0xE0 && c1 < 0xA0) return false;
            if (c == 0xED && c1 >= 0xA0) return false;
            i += 3;
        } else if (c < 0xF5) {
            if (i + 3 >= len) return false;
            uint8_t c1 = static_cast<uint8_t>(data[i + 1]);
            uint8_t c2 = static_cast<uint8_t>(data[i + 2]);
            uint8_t c3 = static_cast<uint8_t>(data[i + 3]);
            if ((c1 & 0xC0) != 0x80) return false;
            if ((c2 & 0xC0) != 0x80) return false;
            if ((c3 & 0xC0) != 0x80) return false;
            if (c == 0xF0 && c1 < 0x90) return false;
            if (c == 0xF4 && c1 >= 0x90) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

static bool _has_text_extension(const luisa::filesystem::path &path) {
    luisa::string ext = path.extension().string().c_str();
    luisa::string fname = path.filename().string().c_str();
    if (!ext.empty()) {
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    }
    static const luisa::unordered_set<luisa::string> text_exts = {
        ".txt",
        ".md",
        ".rst",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cc",
        ".cxx",
        ".c++",
        ".inl",
        ".inc",
        ".py",
        ".pyw",
        ".pyi",
        ".java",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".mjs",
        ".cjs",
        ".json",
        ".jsonc",
        ".json5",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".html",
        ".htm",
        ".xhtml",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".styl",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
        ".ini",
        ".cfg",
        ".conf",
        ".config",
        ".env",
        ".properties",
        ".log",
        ".csv",
        ".tsv",
        ".tex",
        ".latex",
        ".bib",
        ".lua",
        ".rb",
        ".gemspec",
        ".rake",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sc",
        ".r",
        ".m",
        ".mm",
        ".pl",
        ".pm",
        ".php",
        ".phtml",
        ".asp",
        ".aspx",
        ".jsp",
        ".vue",
        ".svelte",
        ".astro",
        ".sql",
        ".ddl",
        ".dml",
        ".cmake",
        ".make",
        ".mk",
        ".dockerfile",
        ".containerfile",
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".gitconfig",
        ".editorconfig",
        ".eslintrc",
        ".prettierrc",
        ".babelrc",
        ".stylelintrc",
        ".npmignore",
        ".nvmrc",
        ".node-version",
        ".python-version",
        ".ruby-version",
        ".htaccess",
        ".nginx",
        ".apache",
        ".awk",
        ".sed",
        ".man",
        ".org",
        ".adoc",
        ".asciidoc",
        ".pod",
        ".rd",
        ".gradle",
        ".sbt",
        ".clj",
        ".cljs",
        ".erl",
        ".hrl",
        ".ex",
        ".exs",
        ".fs",
        ".fsx",
        ".ml",
        ".mli",
        ".hs",
        ".lhs",
        ".jl",
        ".nim",
        ".d",
        ".cr",
        ".v",
        ".sv",
        ".svh",
        ".vhd",
        ".vhdl",
        ".tf",
        ".tfvars",
        ".hcl",
        ".proto",
        ".capnp",
        ".thrift",
        ".graphql",
        ".gql",
        ".prisma",
        ".dart",
        ".groovy",
        ".jenkinsfile",
        ".pp",
        ".puppet",
        ".wdl",
        ".cwl",
        ".snakemake",
        ".smk",
        ".nextflow",
        ".nf",
        ".cs",
        ".csx",
        ".vb",
        ".fsi",
        ".nuspec",
        ".csproj",
        ".vbproj",
        ".sln",
        ".xaml",
        ".resx",
        ".plist",
        ".storyboard",
        ".xcconfig",
        ".ipynb",
        ".rmd",
        ".rnw",
        ".roxygen",
        ".sage",
        ".pyx",
        ".pxd",
        ".pxi",
        ".tcl",
        ".tk",
        ".cob",
        ".cbl",
        ".cpy",
        ".f",
        ".for",
        ".f90",
        ".f95",
        ".f03",
        ".f08",
        ".pas",
        ".pp",
        ".dpr",
        ".lpr",
        ".ada",
        ".adb",
        ".ads",
        ".zig",
        ".zon",
        ".vsh",
        ".fsh",
        ".glsl",
        ".hlsl",
        ".cg",
        ".cginc",
        ".shader",
        ".compute",
        ".cu",
        ".cuh",
        ".hip",
        ".cl",
        ".metal",
        ".ispc",
        ".isph",
        ".ll",
        ".bc",
        ".mlir",
        ".asm",
        ".s",
        ".nasm",
        ".masm",
        ".qml",
        ".qbs",
        ".pro",
        ".pri",
        ".sas",
        ".sps",
        ".thy",
        ".sml",
        ".sig",
        ".fun",
        ".cm",
        ".rkt",
        ".ss",
        ".scm",
        ".sch",
        ".sls",
        ".sld",
        ".lsp",
        ".lisp",
        ".el",
        ".emacs",
        ".vim",
        ".eex",
        ".leex",
        ".heex",
        ".elm",
        ".pure",
        ".idris",
        ".idr",
        ".lidr",
        ".agda",
        ".lagda",
        ".coq",
        ".lean",
        ".lean4",
        ".hlean",
        ".rules",
        ".bzl",
        ".bazelrc",
        ".bazelversion",
        ".star",
        ".gn",
        ".gni",
        ".mojom",
        ".idl",
        ".midl",
        ".fbs",
        ".flatbuffers",
        ".pbtxt",
        ".textproto",
        ".graphqls",
        ".sdl",
        ".schema",
        ".buck",
        ".buckconfig",
        ".fx",
        ".repo",
        ".manifest",
        ".mf",
        ".pom",
        ".ivy",
        ".pubxml",
        ".suo",
        ".user",
        ".axaml",
        ".resw",
        ".resources",
        ".strings",
        ".pbxproj",
        ".podspec",
        ".podfile",
        ".cartfile",
        ".gemfile",
        ".rakefile",
        ".irbrc",
        ".pryrc",
        ".ruby-gemset",
        ".gemrc",
        ".railsrc",
        ".simplecov",
        ".yardopts",
        ".phpunit",
        ".phpcs",
        ".php_cs",
        ".phpstan",
        ".psalm",
        ".composer",
        ".hhconfig",
        ".drupal",
        ".theme",
        ".install",
        ".profile",
        ".perl",
        ".t",
        ".al",
        ".cgi",
        ".plx",
        ".xs",
        ".xsc",
        ".di",
        ".ddoc",
        ".sdl",
        ".purs",
        ".dhall",
        ".nix",
        ".nixexpr",
        ".cabal",
        ".project",
        ".freeze",
        ".hpack",
        ".package.yaml",
        ".stack.yaml",
        ".opam",
        ".dune",
        ".dune-project",
        ".jbuild",
        ".merlin",
        ".mll",
        ".mly",
        ".mlpack",
        ".odocl",
        ".elmx",
        ".ports",
        ".ipkg",
        ".ibc",
        ".agda-lib",
        ".olean",
        ".ML",
        ".grm",
        ".rprofile",
        ".renviron",
        ".i",
        ".ii",
        ".mi",
        ".mii",
        ".opt",
        ".air",
        ".vert",
        ".frag",
        ".geom",
        ".tesc",
        ".tese",
        ".comp",
        ".rgen",
        ".rmiss",
        ".rchit",
        ".rahit",
        ".rint",
        ".rcall",
        ".mak",
        ".nmake",
        ".gmake",
        ".dockerignore",
        ".npmrc",
        ".yarnrc",
        ".pnpmfile",
        ".jshintrc",
        ".jscsrc",
        ".jsbeautifyrc",
    };
    if (!ext.empty() && text_exts.contains(ext)) return true;

    static const luisa::unordered_set<luisa::string> known_names = {
        "Makefile",
        "makefile",
        "GNUmakefile",
        "CMakeLists.txt",
        "Dockerfile",
        "dockerfile",
        "Containerfile",
        "containerfile",
        "Dockerfile.prod",
        "Dockerfile.dev",
        "docker-compose.yml",
        "docker-compose.yaml",
        "README",
        "README.md",
        "README.txt",
        "README.rst",
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        "CHANGELOG",
        "CHANGELOG.md",
        "CHANGELOG.txt",
        "AUTHORS",
        "AUTHORS.txt",
        "CONTRIBUTORS",
        "CONTRIBUTORS.txt",
        "TODO",
        "TODO.txt",
        "NOTES",
        "NOTES.txt",
        "INSTALL",
        "INSTALL.txt",
        "COPYING",
        "COPYING.txt",
        "NEWS",
        "NEWS.txt",
        "CITATIONS",
        "CODE_OF_CONDUCT",
        "CONTRIBUTING",
        "SECURITY",
        "SUPPORT",
        "FAQ",
        "Jenkinsfile",
        "jenkinsfile",
        "Vagrantfile",
        "Rakefile",
        "Gemfile",
        "Guardfile",
        "Berksfile",
        "Capfile",
        "Cheffile",
        "Doxyfile",
        "Procfile",
        "Procfile.dev",
        "build.gradle",
        "settings.gradle",
        "gradlew",
        "gradlew.bat",
        "pom.xml",
        "build.xml",
        "ivy.xml",
        "CMakeCache.txt",
        "configure",
        "configure.ac",
        "Makefile.am",
        "Makefile.in",
        "aclocal.m4",
        "config.h.in",
        "meson.build",
        "meson_options.txt",
        "BUILD",
        "BUILD.bazel",
        "WORKSPACE",
        "WORKSPACE.bazel",
        "SConstruct",
        "SConscript",
        "wscript",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "Pipfile.lock",
        "tox.ini",
        "pytest.ini",
        "Cargo.toml",
        "Cargo.lock",
        "rustfmt.toml",
        "clippy.toml",
        "stack.yaml",
        "package.yaml",
        "cabal.project",
        "cabal.config",
        "pubspec.yaml",
        "pubspec.lock",
        "analysis_options.yaml",
        "Package.swift",
        "Package.resolved",
        "go.mod",
        "go.sum",
        "go.work",
        "composer.json",
        "composer.lock",
        "config.ru",
        "build.boot",
        "build.cljs",
        "deps.edn",
        "shadow-cljs.edn",
        "project.clj",
        "Info.plist",
        "Entitlements.plist",
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".gitconfig",
        ".gitkeep",
        ".editorconfig",
        ".env",
        ".env.local",
        ".env.example",
        ".eslintrc",
        ".eslintrc.json",
        ".eslintrc.js",
        ".eslintrc.yaml",
        ".eslintrc.yml",
        ".prettierrc",
        ".prettierrc.json",
        ".prettierrc.js",
        ".prettierrc.yaml",
        ".prettierrc.yml",
        ".prettierrc.toml",
        ".babelrc",
        ".babelrc.json",
        ".babelrc.js",
        ".postcssrc",
        ".postcssrc.json",
        ".postcssrc.js",
        ".postcssrc.yaml",
        ".postcssrc.yml",
        ".stylelintrc",
        ".stylelintrc.json",
        ".stylelintrc.js",
        ".stylelintrc.yaml",
        ".stylelintrc.yml",
        ".npmignore",
        ".nvmrc",
        ".node-version",
        ".python-version",
        ".ruby-version",
        ".tool-versions",
        ".htaccess",
        ".htpasswd",
        ".yardopts",
        ".rspec",
        ".simplecov",
        ".pryrc",
        ".irbrc",
        ".railsrc",
        ".bashrc",
        ".bash_profile",
        ".bash_login",
        ".bash_logout",
        ".profile",
        ".zshrc",
        ".zprofile",
        ".zlogin",
        ".zlogout",
        ".vimrc",
        ".gvimrc",
        ".exrc",
        ".nvimrc",
        ".ideavimrc",
        ".tmux.conf",
        ".screenrc",
        ".inputrc",
        ".wgetrc",
        ".curlrc",
        ".netrc",
        ".my.cnf",
        ".Xresources",
        ".Xdefaults",
        ".xinitrc",
        ".xsession",
        ".xsessionrc",
        ".mailcap",
        ".mime.types",
        ".muttrc",
        ".fetchmailrc",
        ".procmailrc",
        ".quiltrc",
        ".ackrc",
        ".agignore",
        ".rgignore",
        ".fdignore",
        ".ignore",
        ".dockerignore",
        ".eslintignore",
        ".prettierignore",
        ".stylelintignore",
        ".coffeelintignore",
        ".gruntfile",
        ".gulpfile",
        ".clang-format",
        ".clang-tidy",
        ".clangd",
        ".luacheckrc",
        ".stylua.toml",
        ".selene.toml",
        ".luarc.json",
        ".scalafmt.conf",
        ".scalariform.conf",
        ".scalafix.conf",
        ".scala-steward.conf",
        ".kotlin-version",
        ".ktlint",
        ".detekt",
        ".swift-version",
        ".swiftformat",
        ".swiftlint.yml",
        ".golangci.yml",
        ".golangci.yaml",
        ".go-version",
        ".rustfmt.toml",
        ".rust-toolchain",
        ".rust-toolchain.toml",
        ".perltidyrc",
        ".perlcriticrc",
        ".pylintrc",
        ".flake8",
        ".black",
        ".isort.cfg",
        ".mypy.ini",
        "mypy.ini",
        ".bandit",
        ".coveragerc",
        ".dir-locals.el",
        ".projectile",
        ".ccls",
        ".ccls-root",
        "compile_commands.json",
        ".astylerc",
        ".uncrustify.cfg",
        ".indent.pro",
        ".git-blame-ignore",
        ".git-for-windows-updater",
        ".git-prompt.sh",
        ".mailmap",
        ".mention-bot",
        ".pullapprove.yml",
        ".mergify.yml",
        ".dependabot.yml",
        ".renovaterc",
        ".renovaterc.json",
    };
    return known_names.contains(fname);
}

bool FileBuilder::_is_text_file(const luisa::filesystem::path &path) const {
    if (!_has_text_extension(path)) return false;

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char buf[8192];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        auto n = f.gcount();
        if (!_is_valid_utf8(buf, static_cast<size_t>(n))) {
            return false;
        }
    }
    return true;
}

luisa::string FileBuilder::_hash_file(const luisa::filesystem::path &path) const {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    if (size < 0) return {};
    f.seekg(0, std::ios::beg);
    luisa::vector<char> buffer(static_cast<size_t>(size));
    if (!f.read(buffer.data(), size)) return {};
    vstd::MD5 md5{vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(buffer.data()), buffer.size())};
    return md5.to_string(false);
}

luisa::string FileBuilder::_compute_fingerprint(luisa::string_view text) const {
    luisa::vector<luisa::string> words;
    size_t start = 0;
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || std::isspace(static_cast<unsigned char>(text[i]))) {
            if (i > start) words.emplace_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    luisa::sort(words.begin(), words.end());
    luisa::string fp;
    for (auto &w : words) {
        fp += w;
        fp += '|';
    }
    return fp;
}

void FileBuilder::_scan_and_build() {
    luisa::vector<std::pair<luisa::string, luisa::filesystem::path>> files;
    std::error_code ec;
    for (const auto &root : _paths) {
        LUISA_INFO("FileBuilder scanning root: {}", luisa::to_string(root));
        if (!luisa::filesystem::exists(root, ec) || ec) {
            LUISA_WARNING("Root does not exist: {}", luisa::to_string(root));
            continue;
        }
        if (luisa::filesystem::is_regular_file(root, ec) && !ec) {
            luisa::string abs_path = luisa::to_string(luisa::filesystem::canonical(root, ec));
            if (ec) abs_path = luisa::to_string(luisa::filesystem::absolute(root));
            files.emplace_back(std::move(abs_path), root);
            continue;
        }
        luisa::filesystem::recursive_directory_iterator iter(root, ec);
        if (ec) {
            LUISA_WARNING("Failed to create directory iterator for {}: {}", luisa::to_string(root), ec.message());
            continue;
        }
        for (; iter != luisa::filesystem::recursive_directory_iterator{}; iter.increment(ec)) {
            if (ec) {
                ec.clear();
                continue;
            }
            const auto &entry = *iter;
            if (entry.is_regular_file(ec) && !ec) {
                luisa::string abs_path = luisa::to_string(luisa::filesystem::canonical(entry.path(), ec));
                if (ec) abs_path = luisa::to_string(luisa::filesystem::absolute(entry.path()));
                files.emplace_back(std::move(abs_path), entry.path());
            }
        }
    }
    LUISA_INFO("FileBuilder collected {} files", files.size());

    if (files.empty()) {
        _index = luisa::make_unique<InvertedIndex>();
        return;
    }

    // Parallel hash + text filter
    luisa::vector<luisa::string> paths;
    luisa::vector<luisa::string> hashes;
    luisa::fiber::mutex hash_mtx;

    luisa::fiber::parallel(static_cast<uint32_t>(files.size()), [&](uint32_t i) noexcept {
        const auto &path_str = files[i].first;
        const auto &path = files[i].second;
        if (!const_cast<FileBuilder *>(this)->_is_text_file(path)) return;
        auto h = const_cast<FileBuilder *>(this)->_hash_file(path);
        if (h.empty()) return;
        luisa::fiber::lock lck(hash_mtx);
        paths.push_back(path_str);
        hashes.push_back(std::move(h));
    });

    _file_hashes.clear();
    for (size_t i = 0; i < paths.size(); ++i) {
        _file_hashes.emplace(std::move(paths[i]), std::move(hashes[i]));
    }
    LUISA_INFO("FileBuilder text files: {}", _file_hashes.size());

    // Parallel line processing per file
    struct FileLines {
        luisa::string path;
        luisa::vector<LineEntry> lines;
    };
    luisa::vector<FileLines> all_file_lines;
    luisa::fiber::mutex lines_mtx;

    luisa::fiber::parallel(static_cast<uint32_t>(files.size()), [&](uint32_t i) noexcept {
        const auto &path_str = files[i].first;
        const auto &path = files[i].second;
        auto it = const_cast<FileBuilder *>(this)->_file_hashes.find(path_str);
        if (it == const_cast<FileBuilder *>(this)->_file_hashes.end()) return;

        std::ifstream f(path);
        if (!f) return;
        FileLines fl;
        fl.path = path_str;
        luisa::string line;
        int line_idx = 0;
        while (std::getline(f, line)) {
            luisa::string stripped;
            size_t b = 0;
            while (b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) ++b;
            size_t e = line.size();
            while (e > b && std::isspace(static_cast<unsigned char>(line[e - 1]))) --e;
            if (b >= e) {
                ++line_idx;
                continue;
            }
            stripped = line.substr(b, e - b);
            LineEntry entry;
            entry.path = path_str;
            entry.line_idx = line_idx;
            entry.content = std::move(stripped);
            luisa::string query_text = path_str + ": " + entry.content;
            entry.tokens = const_cast<FileBuilder *>(this)->_tokenizer.tokenize(query_text);
            entry.simhash = SimHash(entry.content).value();
            entry.fingerprint = const_cast<FileBuilder *>(this)->_compute_fingerprint(entry.content);
            fl.lines.push_back(std::move(entry));
            ++line_idx;
        }
        {
            luisa::fiber::lock lck(lines_mtx);
            all_file_lines.push_back(std::move(fl));
        }
    });

    // Flatten lines deterministically
    luisa::vector<LineEntry> all_lines;
    luisa::sort(all_file_lines.begin(), all_file_lines.end(), [](const auto &a, const auto &b) {
        return a.path < b.path;
    });
    for (auto &fl : all_file_lines) {
        for (auto &ln : fl.lines) {
            all_lines.push_back(std::move(ln));
        }
    }

    LUISA_INFO("FileBuilder total lines: {}", all_lines.size());
    _build_index(all_lines);
}

void FileBuilder::_build_index(const luisa::vector<LineEntry> &lines) {
    SimHashLSH lsh(64, 16);
    luisa::unordered_set<luisa::string> seen_fp;
    luisa::vector<LineEntry> deduped;
    deduped.reserve(lines.size());

    int doc_id = 0;
    for (const auto &ln : lines) {
        bool is_dup = false;
        if (seen_fp.contains(ln.fingerprint)) {
            is_dup = true;
        } else {
            SimHash h;
            {
                SimHash tmp(ln.content);
                h = tmp;
            }
            auto cands = lsh.candidates(h);
            for (int other_id : cands) {
                SimHash other;
                {
                    SimHash tmp(deduped[other_id].content);
                    other = tmp;
                }
                if (h.is_near_duplicate(other, 3)) {
                    is_dup = true;
                    break;
                }
            }
        }
        if (is_dup) continue;

        SimHash h(ln.content);
        lsh.add(doc_id, h);
        seen_fp.insert(ln.fingerprint);
        deduped.push_back(ln);
        ++doc_id;
    }

    auto index = luisa::make_unique<InvertedIndex>();
    _doc_info.clear();
    _doc_info.reserve(deduped.size());

    for (size_t i = 0; i < deduped.size(); ++i) {
        index->add_document(static_cast<int>(i), deduped[i].tokens);
        _doc_info.push_back({deduped[i].path, deduped[i].line_idx, deduped[i].content});
    }

    LUISA_INFO("FileBuilder deduped lines: {}", deduped.size());
    if (!deduped.empty()) {
        index->finalize(1.0);
    }
    _index = std::move(index);
    _searcher = luisa::make_unique<Searcher>(*_index, &_tokenizer, _k1, _b);
    LUISA_INFO("FileBuilder index N={}", _index->N());
}

void FileBuilder::_save_cache() {
    if (!_index) return;
    luisa::filesystem::create_directories(_cache_path.parent_path());
    {
        std::ofstream f(_cache_path, std::ios::binary);
        if (!f) return;
        auto write_u8 = [&](uint8_t v) { f.write(reinterpret_cast<const char *>(&v), 1); };
        auto write_u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };

        f.write(reinterpret_cast<const char *>(CACHE_MAGIC.data()), 4);
        write_u8(CACHE_VERSION);
        write_u32(static_cast<uint32_t>(_file_hashes.size()));
        for (const auto &[path, h] : _file_hashes) {
            write_u16(static_cast<uint16_t>(path.size()));
            f.write(path.data(), static_cast<std::streamsize>(path.size()));
            write_u16(static_cast<uint16_t>(h.size()));
            f.write(h.data(), static_cast<std::streamsize>(h.size()));
        }
        write_u32(static_cast<uint32_t>(_doc_info.size()));
        for (const auto &info : _doc_info) {
            write_u16(static_cast<uint16_t>(info.path.size()));
            f.write(info.path.data(), static_cast<std::streamsize>(info.path.size()));
            int32_t li = info.line_index;
            f.write(reinterpret_cast<const char *>(&li), sizeof(li));
            write_u32(static_cast<uint32_t>(info.content.size()));
            f.write(info.content.data(), static_cast<std::streamsize>(info.content.size()));
        }
    }
    _index->save(_index_path, true);
}

bool FileBuilder::_load_cache() {
    std::ifstream f(_cache_path, std::ios::binary);
    if (!f) return false;

    auto read_u8 = [&]() -> uint8_t {
        uint8_t v;
        f.read(reinterpret_cast<char *>(&v), 1);
        return v;
    };
    auto read_u16 = [&]() -> uint16_t {
        uint16_t v;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        return v;
    };
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        return v;
    };

    std::array<uint8_t, 4> magic{};
    f.read(reinterpret_cast<char *>(magic.data()), 4);
    if (magic != CACHE_MAGIC) return false;
    uint8_t version = read_u8();
    if (version != CACHE_VERSION) return false;

    uint32_t num_hashes = read_u32();
    _file_hashes.clear();
    for (uint32_t i = 0; i < num_hashes; ++i) {
        uint16_t path_len = read_u16();
        luisa::string path(path_len, '\0');
        f.read(path.data(), path_len);
        uint16_t h_len = read_u16();
        luisa::string h(h_len, '\0');
        f.read(h.data(), h_len);
        _file_hashes.emplace(std::move(path), std::move(h));
    }

    uint32_t num_docs = read_u32();
    _doc_info.resize(num_docs);
    for (uint32_t i = 0; i < num_docs; ++i) {
        uint16_t path_len = read_u16();
        _doc_info[i].path.resize(path_len);
        f.read(_doc_info[i].path.data(), path_len);
        int32_t li;
        f.read(reinterpret_cast<char *>(&li), sizeof(li));
        _doc_info[i].line_index = li;
        uint32_t content_len = read_u32();
        _doc_info[i].content.resize(content_len);
        f.read(_doc_info[i].content.data(), content_len);
    }

    auto index = luisa::make_unique<InvertedIndex>();
    index->load(_index_path);
    _index = std::move(index);
    _searcher = luisa::make_unique<Searcher>(*_index, &_tokenizer, _k1, _b);
    return true;
}

bool FileBuilder::_cache_valid() const {
    if (!luisa::filesystem::exists(_cache_path)) return false;
    // Validate hashes by re-scanning files quickly (check mtimes first, then hashes)
    // For simplicity, just check existence and that all hashed files still exist
    for (const auto &[path, h] : _file_hashes) {
        auto p = luisa::filesystem::path(path);
        if (!luisa::filesystem::exists(p)) return false;
    }
    // Also check no new files at top level (simplified: always rebuild if paths changed)
    return true;
}

void FileBuilder::update() {
    _scan_and_build();
    _save_cache();
}

luisa::vector<FileBuilderResult> FileBuilder::search(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream,
    luisa::string_view keywords,
    int top_k,
    bool diversify,
    double diversity_lambda,
    bool use_spelling,
    bool use_stemming,
    bool use_string_similarity,
    bool use_adaptive_scoring) {

    if (!_searcher || _index->N() == 0) return {};

    luisa::string query = luisa::string{keywords};
    luisa::string original_query = query;

    // Stemming
    if (use_stemming) {
        luisa::vector<luisa::string> words;
        size_t start = 0;
        for (size_t i = 0; i <= query.size(); ++i) {
            if (i == query.size() || std::isspace(static_cast<unsigned char>(query[i]))) {
                if (i > start) words.push_back(porter_stem(query.substr(start, i - start)));
                start = i + 1;
            }
        }
        query.clear();
        for (size_t i = 0; i < words.size(); ++i) {
            if (i > 0) query += ' ';
            query += words[i];
        }
    }

    // Spelling correction using LevenshteinAutomaton on index terms
    if (use_spelling) {
        auto terms = _index->terms();
        luisa::vector<luisa::string> words;
        size_t start = 0;
        for (size_t i = 0; i <= query.size(); ++i) {
            if (i == query.size() || std::isspace(static_cast<unsigned char>(query[i]))) {
                if (i > start) words.push_back(query.substr(start, i - start));
                start = i + 1;
            }
        }
        bool changed = false;
        for (auto &w : words) {
            if (_index->has_term(w)) continue;
            int max_edits = LevenshteinAutomaton::auto_fuzziness(w);
            if (max_edits == 0) continue;
            LevenshteinAutomaton la(w, max_edits);
            auto matches = la.match(*_index, 1);
            if (!matches.empty()) {
                w = matches[0];
                changed = true;
            }
        }
        if (changed) {
            query.clear();
            for (size_t i = 0; i < words.size(); ++i) {
                if (i > 0) query += ' ';
                query += words[i];
            }
        }
    }

    auto raw_results = _searcher->search(device, stream, query, top_k * 4);
    if (raw_results.empty()) return {};

    size_t n = raw_results.size();
    luisa::vector<double> bm25_scores(n);
    for (size_t i = 0; i < n; ++i) bm25_scores[i] = raw_results[i].second;
    double max_bm25 = 1.0, min_bm25 = 0.0;
    for (double s : bm25_scores) {
        max_bm25 = std::max(max_bm25, s);
        min_bm25 = std::min(min_bm25, s);
    }
    double range = max_bm25 - min_bm25;
    luisa::vector<double> bm25_norm(n, 1.0);
    if (range > 0) {
        for (size_t i = 0; i < n; ++i) {
            bm25_norm[i] = (bm25_scores[i] - min_bm25) / range;
        }
    }

    luisa::vector<double> string_scores(n, 0.0);
    if (use_string_similarity) {
        for (size_t i = 0; i < n; ++i) {
            int doc_id = raw_results[i].first;
            const auto &content = _doc_info[doc_id].content;
            double jw = jaro_winkler_similarity(original_query, content);
            double dice = sorensen_dice_coefficient(original_query, content);
            double ngo = ngram_overlap(original_query, content);
            string_scores[i] = (jw + dice + ngo) / 3.0;
        }
    }

    double bm25_weight = 0.5;
    if (use_adaptive_scoring) {
        BM25Scorer scorer(*_index, _k1, _b);
        QueryPerformancePredictor qpp(*_index, scorer);
        auto qtokens = _tokenizer.tokenize(query);
        if (!qtokens.empty()) {
            if (qpp.is_hard_query(qtokens, 2.0)) {
                bm25_weight = 0.7;
            } else {
                bm25_weight = 0.5;
            }
        }
    }

    luisa::vector<std::pair<int, double>> scored;
    scored.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double final_score = bm25_weight * bm25_norm[i] + (1.0 - bm25_weight) * string_scores[i];
        scored.emplace_back(raw_results[i].first, final_score);
    }
    luisa::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    if (diversify && scored.size() > static_cast<size_t>(top_k)) {
        scored = mmr_rerank(scored, *_index, diversity_lambda, top_k);
    } else if (scored.size() > static_cast<size_t>(top_k)) {
        scored.resize(top_k);
    }

    luisa::vector<FileBuilderResult> results;
    results.reserve(scored.size());
    for (const auto &[doc_id, score] : scored) {
        const auto &info = _doc_info[doc_id];
        results.push_back({doc_id, score, info.path, info.line_index + 1});
    }
    return results;
}

}// namespace tokenize
