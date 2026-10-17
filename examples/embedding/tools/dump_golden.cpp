// bge-m3 embedding engine — activation dump / layer-parity dev tool
// (embedding_model_plan.md §9.1 D4 "activation dump harness", §6 M2/M6 tooling;
// target example_embed_dump_golden). Encodes ONE sentence through the Session
// and writes the per-layer activations observed by the Executor layer trace to
// raw fp32 files next to a small manifest, optionally comparing them against
// the pinned golden layer states (src/tests/data/embedding/golden_layers.bin +
// golden_meta.json). This is the "which of the 24 layers diverged" bisect tool.
//
// CLI:
//   dump_golden <backend> <model.bgm> <tokenizer.json> <sentence_index|text>
//               [--layers [0,3,24]] [--compare <golden-dir>]
// - <sentence_index|text>: a pure-digit argument indexes corpus.txt located
//   next to <tokenizer.json> (lines escape \ \\n \\r — unescaped here);
//   anything else is used as the literal UTF-8 sentence.
// - --layers without a list dumps all 25 traced states; with a comma list it
//   dumps only those layer numbers (trace indexing per executor.h: 0 =
//   embedding output, 1..24 = post-LN2 of encoder layer k-1 — exactly the
//   golden_layers.bin indexing).
// - --compare requires the sentence index to be 0 or 3 (the goldens pin those
//   two sentences) and prints per-layer max-abs-diff against golden_layers.bin.
// - Outputs land in the CWD: dump_layer_NN.bin (rows*cols fp32 LE),
//   dump_dense.bin (the L2-normalized dense vector), dump_meta.json.
//
// Trace wiring (executor.h contract, B1): the callback is invoked on the caller
// host thread after the encode completes, with an immutable slot BufferView
// plus (rows, cols), and consumers may issue their own
// `stream << activations.copy_to(host_span) << synchronize()` — which is what
// this tool does on a dedicated read stream. The hook is reached through a
// duck-typed scan of the Session API (session.set_layer_trace /
// session.executor().set_layer_trace + fire_trace_callbacks / pointer form), so
// the tool compiles against whichever passthrough B1 exposes; with none it
// degrades to a dense-only dump and says so loudly (the parent wires the trace
// at integration).
//
// When embedding/embed/session.h is not in the tree yet (B1 wave in progress)
// the tool builds as a pre-check stub: it validates its arguments and reports
// "SESSION UNAVAILABLE" with exit code 2. Exit codes: 0 ok, 1 bad args /
// missing files / session failure / compare index outside {0,3}, 2 stub build.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace {

    struct Options {
        luisa::string backend;
        std::filesystem::path bgm;
        std::filesystem::path tokenizer_json;
        luisa::string sentence_arg;
        luisa::vector<uint32_t> layers;// empty -> all 25
        std::filesystem::path compare_dir;
    };

    void print_usage() {
        LUISA_INFO("usage: dump_golden <backend> <model.bgm> <tokenizer.json> <sentence_index|text> "
                   "[--layers [a,b,...]] [--compare <golden-dir>]");
    }

    // consumes argv into Options; false + message on malformed input
    bool parse_args(int argc, char *argv[], Options &opt) {
        const auto consume_layer_list = [&opt](std::string_view csv) {
            luisa::vector<uint32_t> parsed;
            uint32_t layer = 0u;
            bool digit_seen = false;
            const auto flush = [&]() {
                parsed.push_back(layer);
                layer = 0u;
                digit_seen = false;
            };
            for(const auto c : csv) {
                if(c >= '0' && c <= '9') {
                    layer = layer * 10u + static_cast<uint32_t>(c - '0');
                    digit_seen = true;
                } else if(c == ',') {
                    flush();
                } else {
                    return false;
                }
            }
            if(digit_seen || !parsed.empty()) {
                flush();
            }
            if(parsed.empty()) {
                return false;
            }
            opt.layers = std::move(parsed);
            return true;
        };
        const auto positional = [&](std::string_view a) {
            if(opt.backend.empty()) {
                opt.backend = luisa::string{a.data(), a.size()};
            } else if(opt.bgm.empty()) {
                opt.bgm = std::filesystem::path{std::string_view{a}};
            } else if(opt.tokenizer_json.empty()) {
                opt.tokenizer_json = std::filesystem::path{std::string_view{a}};
            } else if(opt.sentence_arg.empty()) {
                opt.sentence_arg = luisa::string{a.data(), a.size()};
            } else {
                return false;
            }
            return true;
        };
        for(int i = 1; i < argc; i++) {
            const std::string_view a{argv[i]};
            if(a == "--layers") {
                // optional value: the next argument if it looks like a layer list
                if(i + 1 < argc && argv[i + 1][0] != '-') {
                    if(!consume_layer_list(std::string_view{argv[i + 1]})) {
                        LUISA_INFO("dump_golden: bad --layers list '{}'", argv[i + 1]);
                        return false;
                    }
                    i++;
                }
            } else if(a.starts_with("--layers=")) {
                if(!consume_layer_list(a.substr(9))) {
                    LUISA_INFO("dump_golden: bad --layers list '{}'", a);
                    return false;
                }
            } else if(a == "--compare") {
                if(i + 1 >= argc) {
                    LUISA_INFO("dump_golden: --compare requires a directory argument");
                    return false;
                }
                opt.compare_dir = std::filesystem::path{argv[++i]};
            } else if(a.starts_with("-")) {
                LUISA_INFO("dump_golden: unknown option '{}'", a);
                return false;
            } else if(!positional(a)) {
                LUISA_INFO("dump_golden: too many positional arguments ('{}')", a);
                return false;
            }
        }
        if(opt.backend.empty() || opt.bgm.empty() || opt.tokenizer_json.empty() || opt.sentence_arg.empty()) {
            LUISA_INFO("dump_golden: <backend> <model.bgm> <tokenizer.json> <sentence_index|text> are all required");
            return false;
        }
        return true;
    }

    bool arg_is_sentence_index(const luisa::string &a, uint32_t &idx) {
        if(a.empty()) {
            return false;
        }
        uint64_t v = 0ull;
        for(const auto c : a) {
            if(c < '0' || c > '9') {
                return false;
            }
            v = v * 10ull + static_cast<uint64_t>(c - '0');
            if(v > 0xFFFFFFFFull) {
                return false;
            }
        }
        idx = static_cast<uint32_t>(v);
        return true;
    }

    // shared validation of the parsed arguments (both build modes)
    bool validate_files(const Options &opt, luisa::string_view tool) {
        if(!std::filesystem::exists(opt.bgm)) {
            LUISA_INFO("{}: packed model '{}' does not exist", tool, opt.bgm.string());
            return false;
        }
        if(!std::filesystem::exists(opt.tokenizer_json)) {
            LUISA_INFO("{}: tokenizer '{}' does not exist", tool, opt.tokenizer_json.string());
            return false;
        }
        return true;
    }

}// namespace

#if __has_include("embedding/embed/session.h")
// ---- full build: the Session API is present ---------------------------------
#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

#include <luisa/core/stl/functional.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <yyjson.h>

#include "embedding/embed/session.h"

using namespace luisa;

namespace {

    using compute::BufferView;

    // the executor trace-hook contract type (executor.h LayerTraceCallback,
    // re-declared here so this tool does not depend on the alias existing)
    using TraceFn = luisa::function<void(uint32_t layer, BufferView<float> activations, uint32_t rows, uint32_t cols)>;

    template<class S>
    concept SessionLevelTrace = requires(S &s, TraceFn f) { s.set_layer_trace(std::move(f)); };
    template<class S>
    concept ExecutorRefTrace = requires(S &s, TraceFn f) { s.executor().set_layer_trace(std::move(f)); } && requires(S &s) { s.executor().fire_trace_callbacks(); };
    template<class S>
    concept ExecutorPtrTrace = requires(S &s, TraceFn f) { s.executor()->set_layer_trace(std::move(f)); } && requires(S &s) { s.executor()->fire_trace_callbacks(); };

    // corpus.txt lines escape backslash, \n and \r (golden_meta.json
    // "corpus_escapes"); readers must unescape after reading a line.
    luisa::string unescape_corpus_line(std::string_view line) {
        luisa::string out;
        out.reserve(line.size());
        size_t i = 0u;
        while(i < line.size()) {
            if(line[i] == '\\' && i + 1u < line.size()) {
                const auto c = line[i + 1u];
                if(c == 'n') {
                    out.push_back('\n');
                    i += 2u;
                    continue;
                }
                if(c == 'r') {
                    out.push_back('\r');
                    i += 2u;
                    continue;
                }
                if(c == '\\') {
                    out.push_back('\\');
                    i += 2u;
                    continue;
                }
            }
            out.push_back(line[i]);
            i++;
        }
        return out;
    }

    luisa::vector<luisa::string> load_corpus(const std::filesystem::path &corpus_file) {
        luisa::vector<luisa::string> lines;
        if(auto *f = std::fopen(corpus_file.string().c_str(), "rb"); f != nullptr) {
            char buf[65536];
            while(std::fgets(buf, sizeof(buf), f) != nullptr) {
                luisa::string line{buf};
                while(!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                    line.pop_back();
                }
                lines.push_back(unescape_corpus_line(line));
            }
            std::fclose(f);
        } else {
            LUISA_ERROR("cannot open corpus file \"{}\"", corpus_file.string());
        }
        return lines;
    }

    luisa::string json_escape(std::string_view s) {
        luisa::string out;
        out.reserve(s.size() + 8u);
        for(const auto c : s) {
            switch(c) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if(static_cast<unsigned char>(c) < 0x20u) {
                        out += luisa::format("\\u{:04x}", static_cast<uint32_t>(static_cast<unsigned char>(c)));
                    } else {
                        out.push_back(c);
                    }
            }
        }
        return out;
    }

    void write_floats_file(const std::filesystem::path &p, const luisa::vector<float> &v) {
        if(auto *f = std::fopen(p.string().c_str(), "wb"); f != nullptr) {
            if(!v.empty()) {
                const auto wrote = std::fwrite(v.data(), sizeof(float), v.size(), f);
                if(wrote != v.size()) {
                    std::fclose(f);
                    LUISA_ERROR("short write on \"{}\"", p.string());
                }
            }
            std::fclose(f);
        } else {
            LUISA_ERROR("cannot open \"{}\" for writing", p.string());
        }
    }

    void write_text_file(const std::filesystem::path &p, std::string_view text) {
        if(auto *f = std::fopen(p.string().c_str(), "wb"); f != nullptr) {
            std::fwrite(text.data(), 1u, text.size(), f);
            std::fclose(f);
        } else {
            LUISA_ERROR("cannot open \"{}\" for writing", p.string());
        }
    }

    struct Dump {
        uint32_t layer{};
        uint32_t rows{};
        uint32_t cols{};
        luisa::vector<float> values;
    };

    // collects the traced layer states via host-thread readbacks (the
    // executor.h contract guarantees the callback fires after completion with
    // an immutable slot view; this tool owns its read stream)
    struct TraceCollector {
        compute::Stream *read_stream{nullptr};
        const luisa::vector<uint32_t> *selected{nullptr};// empty -> all
        luisa::vector<Dump> dumps;

        [[nodiscard]] bool is_selected(uint32_t layer) const noexcept {
            return selected == nullptr || selected->empty() ||
                   std::find(selected->begin(), selected->end(), layer) != selected->end();
        }
        TraceFn callback() {
            return [this](uint32_t layer, BufferView<float> view, uint32_t rows, uint32_t cols) noexcept {
                if(!view) {
                    LUISA_WARNING("dump_golden: trace callback for layer {} with an invalid buffer view", layer);
                    return;
                }
                if(!is_selected(layer) || rows == 0u || cols == 0u) {
                    return;
                }
                const uint64_t want = static_cast<uint64_t>(rows) * cols;
                uint64_t n = want;
                if(static_cast<uint64_t>(view.size()) < want) {
                    LUISA_WARNING("dump_golden: layer {} view holds {} floats < rows*cols {}; dumping the available prefix",
                                  layer, view.size(), want);
                    n = view.size();
                }
                Dump d;
                d.layer = layer;
                d.rows = rows;
                d.cols = cols;
                d.values.resize(static_cast<size_t>(n));
                *read_stream << view.subview(0u, static_cast<size_t>(n)).copy_to(luisa::span{d.values})
                             << compute::synchronize();
                dumps.push_back(std::move(d));
            };
        }
    };

    // reach the executor trace hook through whichever passthrough the Session
    // exposes; wrapped in a class template so the concept branches are
    // dependent (a discarded if-constexpr in a plain function is still
    // semantically checked and would error on missing members).
    template<class S>
    struct TraceWiring {
        static bool wire(S &session, TraceCollector &collector, luisa::function<void(S &)> &fire_out) {
            if constexpr(SessionLevelTrace<S>) {
                session.set_layer_trace(collector.callback());
                fire_out = [](S &) noexcept {}; // encode fires synchronously
                return true;
            } else if constexpr(ExecutorRefTrace<S>) {
                session.executor().set_layer_trace(collector.callback());
                fire_out = [](S &s) noexcept { s.executor().fire_trace_callbacks(); };
                return true;
            } else if constexpr(ExecutorPtrTrace<S>) {
                session.executor()->set_layer_trace(collector.callback());
                fire_out = [](S &s) noexcept { s.executor()->fire_trace_callbacks(); };
                return true;
            } else {
                fire_out = [](S &) noexcept {};
                return false;
            }
        }
    };

    // golden layer states (golden_meta.json "layers" + golden_layers.bin):
    // entries {i, layer, rows, dim, offset_bytes}, dtype float32-le.
    struct GoldenState {
        uint32_t rows{};
        uint32_t dim{};
        uint64_t offset_bytes{};
    };
    struct GoldenLayers {
        uint32_t sentence{};
        luisa::vector<std::pair<uint32_t, GoldenState>> by_layer;
        std::filesystem::path bin;
    };

    struct YyjsonDoc {
        yyjson_doc *doc{nullptr};
        YyjsonDoc() = default;
        ~YyjsonDoc() {
            if(doc != nullptr) {
                yyjson_doc_free(doc);
            }
        }
        YyjsonDoc(const YyjsonDoc &) = delete;
        YyjsonDoc &operator=(const YyjsonDoc &) = delete;
    };

    bool load_golden_layers(const std::filesystem::path &dir, uint32_t sentence, GoldenLayers &out) {
        const auto meta_file = dir / "golden_meta.json";
        YyjsonDoc holder;
        yyjson_read_err rerr;
        holder.doc = yyjson_read_file(meta_file.string().c_str(), 0u, nullptr, &rerr);
        if(holder.doc == nullptr) {
            LUISA_INFO("dump_golden: cannot read golden meta \"{}\": {} (pos {})",
                       meta_file.string(), rerr.msg, rerr.pos);
            return false;
        }
        const auto *root = yyjson_doc_get_root(holder.doc);
        const auto *layers = yyjson_obj_get(root, "layers");
        const auto *file = layers != nullptr ? yyjson_obj_get(layers, "file") : nullptr;
        const auto *entries = layers != nullptr ? yyjson_obj_get(layers, "entries") : nullptr;
        if(!yyjson_is_str(file) || !yyjson_is_arr(entries)) {
            LUISA_INFO("dump_golden: golden_meta.json has no usable 'layers' section");
            return false;
        }
        out.sentence = sentence;
        out.bin = dir / std::filesystem::path{yyjson_get_str(file)};
        size_t i = 0u;
        size_t max = 0u;
        const yyjson_val *e = nullptr;
        yyjson_arr_foreach(entries, i, max, e) {
            if(!yyjson_is_obj(e)) {
                continue;
            }
            const auto *vi = yyjson_obj_get(e, "i");
            const auto *vl = yyjson_obj_get(e, "layer");
            const auto *vr = yyjson_obj_get(e, "rows");
            const auto *vd = yyjson_obj_get(e, "dim");
            const auto *vo = yyjson_obj_get(e, "offset_bytes");
            if(!yyjson_is_uint(vi) || !yyjson_is_uint(vl) || !yyjson_is_uint(vr) ||
               !yyjson_is_uint(vd) || !yyjson_is_uint(vo)) {
                continue;
            }
            if(yyjson_get_uint(vi) == sentence) {
                out.by_layer.emplace_back(
                    static_cast<uint32_t>(yyjson_get_uint(vl)),
                    GoldenState{
                        static_cast<uint32_t>(yyjson_get_uint(vr)),
                        static_cast<uint32_t>(yyjson_get_uint(vd)),
                        static_cast<uint64_t>(yyjson_get_uint(vo))});
            }
        }
        if(out.by_layer.empty()) {
            LUISA_INFO("dump_golden: golden_layers has no states for sentence {}", sentence);
            return false;
        }
        return true;
    }

    // per-layer max-abs-diff of every dumped state against the golden sentence
    int run_compare(const GoldenLayers &golden, const luisa::vector<Dump> &dumps) {
        LUISA_INFO("dump_golden: comparing against golden sentence {} from \"{}\"",
                   golden.sentence, golden.bin.string());
        float overall = 0.f;
        uint32_t compared = 0u;
        for(const auto &d : dumps) {
            const auto it = std::find_if(golden.by_layer.begin(), golden.by_layer.end(),
                                         [&d](const auto &p) { return p.first == d.layer; });
            if(it == golden.by_layer.end()) {
                LUISA_WARNING("dump_golden: no golden state for layer {} (sentence {})", d.layer, golden.sentence);
                continue;
            }
            const auto &gs = it->second;
            if(gs.dim != d.cols) {
                LUISA_WARNING("dump_golden: layer {}: cols {} != golden dim {} — skipped", d.layer, d.cols, gs.dim);
                continue;
            }
            if(gs.rows != d.rows) {
                LUISA_WARNING("dump_golden: layer {}: rows {} != golden rows {} — comparing the common prefix",
                              d.layer, d.rows, gs.rows);
            }
            const size_t count = static_cast<size_t>(std::min(gs.rows, d.rows)) * d.cols;
            if(count == 0ull || count > d.values.size()) {
                LUISA_WARNING("dump_golden: layer {} has too few dumped floats ({} < {})", d.layer, d.values.size(), count);
                continue;
            }
            luisa::vector<float> gv(count);
            if(auto *f = std::fopen(golden.bin.string().c_str(), "rb"); f != nullptr) {
                // golden_layers.bin is ~3.3 MB; 32-bit std::fseek is sufficient
                if(std::fseek(f, static_cast<long>(gs.offset_bytes), SEEK_SET) != 0 ||
                   std::fread(gv.data(), sizeof(float), count, f) != count) {
                    LUISA_INFO("dump_golden: cannot read {} golden floats at offset {} of \"{}\"",
                               count, gs.offset_bytes, golden.bin.string());
                    std::fclose(f);
                    return 1;
                }
                std::fclose(f);
            } else {
                LUISA_INFO("dump_golden: cannot open golden blob file \"{}\"", golden.bin.string());
                return 1;
            }
            float max_diff = 0.f;
            size_t at = 0u;
            for(size_t k = 0u; k < count; k++) {
                const float diff = std::abs(gv[k] - d.values[k]);
                if(diff > max_diff) {
                    max_diff = diff;
                    at = k;
                }
            }
            LUISA_INFO("  layer {:2d}: rows={} cols={} compared={} max_abs_diff={:.6e} (at {}, got {:.8g} vs golden {:.8g})",
                       d.layer, d.rows, d.cols, count, max_diff, at, d.values[at], gv[at]);
            overall = std::max(overall, max_diff);
            compared++;
        }
        LUISA_INFO("dump_golden: {} layer(s) compared, overall max_abs_diff = {:.6e}", compared, overall);
        return 0;
    }

}// namespace

int main(int argc, char *argv[]) {
    Options opt;
    if(!parse_args(argc, argv, opt)) {
        print_usage();
        return 1;
    }
    if(!validate_files(opt, "dump_golden")) {
        return 1;
    }

    // resolve the sentence: numeric -> corpus.txt next to tokenizer.json
    uint32_t sentence_index = 0xFFFFFFFFu;
    const bool is_index = arg_is_sentence_index(opt.sentence_arg, sentence_index);
    luisa::string text = opt.sentence_arg;
    if(is_index) {
        const auto corpus_file = opt.tokenizer_json.parent_path() / "corpus.txt";
        const auto corpus = load_corpus(corpus_file);
        if(sentence_index >= corpus.size()) {
            LUISA_INFO("dump_golden: sentence index {} out of range (corpus has {} lines)",
                       sentence_index, corpus.size());
            return 1;
        }
        text = corpus[sentence_index];
    }
    if(!opt.compare_dir.empty() && !(is_index && (sentence_index == 0u || sentence_index == 3u))) {
        LUISA_INFO("dump_golden: --compare requires a numeric sentence index of 0 or 3 "
                   "(the pinned golden layer states); got '{}'", opt.sentence_arg);
        return 1;
    }

    compute::Context ctx{argv[0]};
    auto device = ctx.create_device(opt.backend);
    if(!device) {
        LUISA_INFO("dump_golden: backend '{}' not available", opt.backend);
        return 1;
    }
    auto session = luisa::embed::Session::create(device, opt.bgm, opt.tokenizer_json);
    if(session == nullptr) {
        LUISA_INFO("dump_golden: Session::create failed for package '{}'", opt.bgm.string());
        return 1;
    }
    auto read_stream = device.create_stream();// tool-owned; never the worker's

    // wire the layer trace through whichever passthrough the Session exposes
    TraceCollector collector;
    collector.read_stream = &read_stream;
    collector.selected = &opt.layers;
    luisa::function<void(luisa::embed::Session &)> fire_trace;
    const bool trace_wired = TraceWiring<luisa::embed::Session>::wire(*session, collector, fire_trace);
    if(!trace_wired) {
        LUISA_WARNING("dump_golden: the Session exposes no layer-trace passthrough "
                      "(set_layer_trace on Session nor executor()/executor()->set_layer_trace + "
                      "fire_trace_callbacks); dumping the dense output only — the parent wires "
                      "the trace at integration");
    }

    luisa::embed::EncodeOptions enc_opt;// max_seq_len = 512, heads off
    const auto embedding = session->encode(std::string_view{text.data(), text.size()}, enc_opt);
    fire_trace(*session);

    // sort dumps by layer for stable manifest/file order
    std::sort(collector.dumps.begin(), collector.dumps.end(),
              [](const Dump &a, const Dump &b) { return a.layer < b.layer; });

    // write dump_layer_NN.bin + dump_dense.bin + dump_meta.json into the CWD
    for(const auto &d : collector.dumps) {
        write_floats_file(std::filesystem::path{luisa::format("dump_layer_{:02d}.bin", d.layer)}, d.values);
    }
    write_floats_file(std::filesystem::path{"dump_dense.bin"}, embedding.dense);
    {
        luisa::string meta = luisa::format(
            "{{\n  \"tool\": \"dump_golden\",\n  \"backend\": \"{}\",\n  \"package\": \"{}\",\n"
            "  \"sentence_index\": {},\n  \"text\": \"{}\",\n  \"token_rows\": {},\n"
            "  \"dense_file\": \"dump_dense.bin\", \"dense_floats\": {},\n  \"layers\": [",
            opt.backend, opt.bgm.string(),
            is_index ? luisa::format("{}", sentence_index) : luisa::string{"null"},
            json_escape(text),
            collector.dumps.empty() ? 0u : collector.dumps.front().rows,
            embedding.dense.size());
        for(size_t i = 0u; i < collector.dumps.size(); i++) {
            const auto &d = collector.dumps[i];
            meta += luisa::format("\n    {{\"layer\": {}, \"rows\": {}, \"cols\": {}, \"floats\": {}, \"file\": \"dump_layer_{:02d}.bin\"}}",
                                  d.layer, d.rows, d.cols, d.values.size(), d.layer);
            if(i + 1u < collector.dumps.size()) {
                meta += ",";
            }
        }
        if(!collector.dumps.empty()) {
            meta += "\n  ";
        }
        meta += "]\n}\n";
        write_text_file(std::filesystem::path{"dump_meta.json"}, meta);
    }
    LUISA_INFO("dump_golden: wrote {} layer dump(s) + dump_dense.bin + dump_meta.json for \"{}\"",
               collector.dumps.size(),
               text.size() > 48u ? luisa::string{text.data(), 48u} + "..." : text);

    if(!opt.compare_dir.empty()) {
        GoldenLayers golden;
        if(!load_golden_layers(opt.compare_dir, sentence_index, golden)) {
            return 1;
        }
        if(collector.dumps.empty()) {
            LUISA_WARNING("dump_golden: nothing was traced — no layer comparison possible");
            return 1;
        }
        return run_compare(golden, collector.dumps);
    }
    return 0;
}
#else
// ---- stub build: B1's session.h is not in the tree yet ----------------------
int main(int argc, char *argv[]) {
    Options opt;
    if(!parse_args(argc, argv, opt)) {
        print_usage();
        return 1;
    }
    if(!validate_files(opt, "dump_golden")) {
        return 1;
    }
    uint32_t sentence_index = 0u;
    const bool is_index = arg_is_sentence_index(opt.sentence_arg, sentence_index);
    LUISA_INFO("dump_golden: arguments accepted (backend '{}', package '{}', tokenizer '{}', "
               "sentence '{}' {}, layers {}, compare '{}')",
               opt.backend, opt.bgm.string(), opt.tokenizer_json.string(), opt.sentence_arg,
               is_index ? luisa::format("(corpus index {})", sentence_index) : luisa::string{"(literal text)"},
               opt.layers.empty() ? "all(25)" : luisa::format("{} selected", opt.layers.size()),
               opt.compare_dir.empty() ? "<none>" : opt.compare_dir.string());
    LUISA_INFO("SESSION UNAVAILABLE: embedding/embed/session.h was not in the tree when this "
               "binary was compiled (B1 executor/session deliverable); no encode/layer dump is "
               "possible. Rebuild example_embed_dump_golden after B1 lands.");
    return 2;
}
#endif
