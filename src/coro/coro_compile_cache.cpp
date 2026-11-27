#include <cstdlib>

#include "coro_compile_cache.h"

#include <luisa/ast/callable_library.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/version.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/xir_interchange.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace luisa::compute::detail {

namespace coro_compile_cache_detail {

constexpr std::array<std::byte, 8u> kCacheMagic{
    std::byte{'L'}, std::byte{'C'}, std::byte{'C'}, std::byte{'O'},
    std::byte{'R'}, std::byte{'O'}, std::byte{'C'}, std::byte{0u}};
constexpr uint32_t kCacheFormatVersion = 1u;
constexpr uint32_t kMaxMetaCount = 1u << 24u;
constexpr size_t kMaxCoroDepth = 64u;
constexpr size_t kMaxAstWalkDepth = 4096u;
constexpr size_t kDefaultXirMaxInsts = 16384u;

struct CoroCacheFileHeader {
    std::array<std::byte, 8u> magic{};
    uint32_t format_version{0u};
    uint32_t flags{0u};
    uint64_t key_hi{0u};
    uint64_t key_lo{0u};
    uint64_t meta_offset{0u};
    uint64_t meta_size{0u};
    uint64_t meta_checksum{0u};
    uint64_t ast_offset{0u};
    uint64_t ast_size{0u};
    uint64_t ast_checksum{0u};
    uint64_t xir_offset{0u};
    uint64_t xir_size{0u};
    uint64_t xir_checksum{0u};
};
static_assert(std::is_trivially_copyable_v<CoroCacheFileHeader>);
static_assert(sizeof(CoroCacheFileHeader) == 104u);

[[nodiscard]] bool env_flag_enabled(const char *name) noexcept {
    if (auto val = luisa::get_environment_variable(name)) {
        return luisa::string_view{*val} == "1";
    }
    return false;
}

[[nodiscard]] bool cache_enabled() noexcept {
    if (auto val = luisa::get_environment_variable("LUISA_CORO_COMPILE_CACHE")) {
        luisa::string_view sv{*val};
        if (sv == "0" || sv == "false" || sv == "FALSE" ||
            sv == "off" || sv == "OFF" || sv == "no" || sv == "NO") {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool any_diagnostic_flag_enabled() noexcept {
    static constexpr std::array flags{
        "LUISA_CORO_VERIFY_PASS_DOMAIN",
        "LUISA_XIR_VERIFY_INTERMEDIATE",
        "LUISA_CORO_VERIFY_DENSE_REACHING_VALUES",
        "LUISA_CORO_VERIFY_ALLOCA_ORDER",
        "LUISA_CORO_VERIFY_HANDLER_SCRATCH_GRAPH",
        "LUISA_XIR_VERIFY_REMAINING_DIVERGENT_INDEX",
        "LUISA_XIR2AST_VERIFY_VALUE_MAP_CHECKPOINTS",
        "LUISA_CORO_VERIFY_DENSE_DATAFLOW",
        "LUISA_CORO_DUMP_FRAME_LAYOUT",
        "LUISA_CORO_DUMP_ALLOCA_SCOPE",
        "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL",
        "LUISA_XIR_TRACE_PASSES",
        "LUISA_XIR_VERIFY_SELECTION_EXIT_RELATION_UPDATES",
        "LUISA_XIR_PROFILE_POINTER_USAGE",
    };
    for (const char *flag : flags) {
        if (env_flag_enabled(flag)) { return true; }
    }
    return false;
}

[[nodiscard]] size_t xir_cache_max_insts() noexcept {
    if (auto val = luisa::get_environment_variable("LUISA_CORO_COMPILE_CACHE_XIR_MAX_INSTS")) {
        size_t parsed = 0u;
        auto sv = luisa::string_view{*val};
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), parsed);
        if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
            return parsed;
        }
    }
    return kDefaultXirMaxInsts;
}

[[nodiscard]] luisa::filesystem::path resolve_cache_dir() noexcept {
    if (auto override_dir = luisa::get_environment_variable("LUISA_CORO_COMPILE_CACHE_DIR");
        override_dir && !override_dir->empty()) {
        return luisa::platform::path_from_narrow(*override_dir);
    }
    auto exe_narrow = luisa::current_executable_path();
    auto exe_path = luisa::platform::path_from_narrow(exe_narrow);
    std::error_code ec;
    if (exe_path.is_relative()) {
        auto abs = luisa::filesystem::absolute(exe_path, ec);
        if (!ec && !abs.empty()) { exe_path = std::move(abs); }
    }
    auto canon = luisa::filesystem::canonical(exe_path, ec);
    if (!ec && !canon.empty()) { exe_path = std::move(canon); }
    luisa::filesystem::path base_dir;
    if (!exe_path.empty() && luisa::filesystem::is_directory(exe_path, ec)) {
        base_dir = exe_path;
    } else if (!exe_path.empty() && exe_path.has_parent_path()) {
        base_dir = exe_path.parent_path();
    } else {
        base_dir = luisa::filesystem::current_path(ec);
    }
    return base_dir / luisa::platform::path_from_narrow(".cache") /
           luisa::platform::path_from_narrow("coro");
}

[[nodiscard]] luisa::filesystem::path make_cache_file_path(
    const luisa::filesystem::path &dir, uint64_t key_hi, uint64_t key_lo) noexcept {
    auto filename = luisa::format("coro_v{}_{:016x}_{:016x}.lcc",
                                  kCacheFormatVersion, key_hi, key_lo);
    return dir / luisa::platform::path_from_narrow(filename);
}

[[nodiscard]] unsigned long current_process_id() noexcept {
#ifdef _WIN32
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(::getpid());
#endif
}

[[nodiscard]] bool read_file_bytes(
    const luisa::filesystem::path &path,
    luisa::vector<std::byte> &out_bytes,
    size_t offset = 0u,
    size_t max_bytes = std::numeric_limits<size_t>::max()) noexcept {
    std::error_code ec;
    if (!luisa::filesystem::is_regular_file(path, ec) || ec) { return false; }
    auto file_size_u64 = luisa::filesystem::file_size(path, ec);
    if (ec || offset > file_size_u64) { return false; }
    auto avail = static_cast<size_t>(file_size_u64 - offset);
    auto to_read = max_bytes == std::numeric_limits<size_t>::max() ? avail : max_bytes;
    if (to_read > avail) { return false; }
    std::ifstream in{path, std::ios::binary};
    if (!in.is_open()) { return false; }
    if (offset != 0u) {
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in.good()) { return false; }
    }
    out_bytes.resize(to_read);
    if (to_read != 0u) {
        in.read(reinterpret_cast<char *>(out_bytes.data()),
                static_cast<std::streamsize>(to_read));
        if (!in.good() || static_cast<size_t>(in.gcount()) != to_read) {
            out_bytes.clear();
            return false;
        }
    }
    return true;
}

class KeyHasher {
    luisa::vector<uint64_t> _words;

public:
    void add_u64(uint64_t v) noexcept { _words.emplace_back(v); }
    void add_str(luisa::string_view s) noexcept {
        add_u64(static_cast<uint64_t>(s.size()));
        add_u64(s.empty() ? uint64_t{0u} : luisa::hash64(s.data(), s.size(), hash64_default_seed));
    }
    void finalize(uint64_t &out_hi, uint64_t &out_lo) const noexcept {
        constexpr uint64_t seed_hi = 0x4c43434f524f4b31ull;
        constexpr uint64_t seed_lo = 0x9e3779b97f4a7c15ull;
        out_hi = luisa::hash64(_words.data(), _words.size() * sizeof(uint64_t), seed_hi);
        out_lo = luisa::hash64(_words.data(), _words.size() * sizeof(uint64_t), seed_lo);
    }
};

[[nodiscard]] bool collect_and_hash_function_tree(
    const FunctionBuilder *fb,
    bool is_root,
    luisa::unordered_set<const FunctionBuilder *> &visited,
    luisa::unordered_set<const FunctionBuilder *> &visiting,
    KeyHasher &hasher,
    size_t fn_depth = 0u) noexcept {
    if (fb == nullptr || !fb->hash_computed()) { return false; }
    if (visited.contains(fb)) { return true; }
    if (visiting.contains(fb) || fn_depth >= kMaxAstWalkDepth) { return false; }
    if (!fb->external_callables().empty()) { return false; }
    static_cast<void>(is_root);

    visiting.emplace(fb);
    luisa::unordered_set<const Expression *> seen_exprs;
    auto walk_expr = [&](auto &self, const Expression *expr, size_t d) noexcept -> bool {
        if (expr == nullptr) { return true; }
        if (d >= kMaxAstWalkDepth) { return false; }
        if (!seen_exprs.emplace(expr).second) { return true; }
        switch (expr->tag()) {
            case Expression::Tag::UNARY:
                return self(self, static_cast<const UnaryExpr *>(expr)->operand(), d + 1u);
            case Expression::Tag::BINARY: {
                auto *b = static_cast<const BinaryExpr *>(expr);
                return self(self, b->lhs(), d + 1u) && self(self, b->rhs(), d + 1u);
            }
            case Expression::Tag::MEMBER:
                return self(self, static_cast<const MemberExpr *>(expr)->self(), d + 1u);
            case Expression::Tag::ACCESS: {
                auto *a = static_cast<const AccessExpr *>(expr);
                return self(self, a->range(), d + 1u) && self(self, a->index(), d + 1u);
            }
            case Expression::Tag::LITERAL:
            case Expression::Tag::REF:
            case Expression::Tag::CONSTANT:
            case Expression::Tag::TYPE_ID:
            case Expression::Tag::STRING_ID:
                return true;
            case Expression::Tag::CALL: {
                auto *c = static_cast<const CallExpr *>(expr);
                if (c->is_external()) { return false; }
                for (auto *arg : c->arguments()) {
                    if (!self(self, arg, d + 1u)) { return false; }
                }
                if (c->is_custom()) {
                    auto callee = c->custom();
                    if (callee.builder() != nullptr &&
                        !collect_and_hash_function_tree(
                            callee.builder(), false, visited, visiting, hasher, fn_depth + 1u)) {
                        return false;
                    }
                }
                return true;
            }
            case Expression::Tag::CAST:
                return self(self, static_cast<const CastExpr *>(expr)->expression(), d + 1u);
            case Expression::Tag::FUNC_REF:
            case Expression::Tag::CPUCUSTOM:
            case Expression::Tag::GPUCUSTOM:
                return false;
        }
        return false;
    };

    auto walk_stmt = [&](auto &self, const Statement *stmt, size_t d) noexcept -> bool {
        if (stmt == nullptr || d >= kMaxAstWalkDepth) { return false; }
        switch (stmt->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::COMMENT:
                return true;
            case Statement::Tag::RETURN:
                return walk_expr(walk_expr, static_cast<const ReturnStmt *>(stmt)->expression(), d + 1u);
            case Statement::Tag::SCOPE: {
                for (auto *s : static_cast<const ScopeStmt *>(stmt)->statements()) {
                    if (!self(self, s, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::IF: {
                auto *ifs = static_cast<const IfStmt *>(stmt);
                return walk_expr(walk_expr, ifs->condition(), d + 1u) &&
                       self(self, ifs->true_branch(), d + 1u) &&
                       self(self, ifs->false_branch(), d + 1u);
            }
            case Statement::Tag::LOOP: {
                auto *ls = static_cast<const LoopStmt *>(stmt);
                return walk_expr(walk_expr, ls->while_condition(), d + 1u) &&
                       self(self, ls->body(), d + 1u);
            }
            case Statement::Tag::EXPR:
                return walk_expr(walk_expr, static_cast<const ExprStmt *>(stmt)->expression(), d + 1u);
            case Statement::Tag::SWITCH: {
                auto *sw = static_cast<const SwitchStmt *>(stmt);
                return walk_expr(walk_expr, sw->expression(), d + 1u) &&
                       self(self, sw->body(), d + 1u);
            }
            case Statement::Tag::SWITCH_CASE:
            case Statement::Tag::SWITCH_CASE_GROUP: {
                auto *sc = static_cast<const SwitchCaseStmt *>(stmt);
                for (auto *e : sc->expressions()) {
                    if (!walk_expr(walk_expr, e, d + 1u)) { return false; }
                }
                return self(self, sc->body(), d + 1u);
            }
            case Statement::Tag::SWITCH_DEFAULT:
                return self(self, static_cast<const SwitchDefaultStmt *>(stmt)->body(), d + 1u);
            case Statement::Tag::ASSIGN: {
                auto *as = static_cast<const AssignStmt *>(stmt);
                return walk_expr(walk_expr, as->lhs(), d + 1u) &&
                       walk_expr(walk_expr, as->rhs(), d + 1u);
            }
            case Statement::Tag::FOR: {
                auto *fs = static_cast<const ForStmt *>(stmt);
                return walk_expr(walk_expr, fs->variable(), d + 1u) &&
                       walk_expr(walk_expr, fs->condition(), d + 1u) &&
                       walk_expr(walk_expr, fs->step(), d + 1u) &&
                       self(self, fs->body(), d + 1u);
            }
            case Statement::Tag::RAY_QUERY: {
                auto *rq = static_cast<const RayQueryStmt *>(stmt);
                return walk_expr(walk_expr, rq->query(), d + 1u) &&
                       self(self, rq->on_triangle_candidate(), d + 1u) &&
                       self(self, rq->on_procedural_candidate(), d + 1u);
            }
            case Statement::Tag::AUTO_DIFF:
                return self(self, static_cast<const AutoDiffStmt *>(stmt)->body(), d + 1u);
            case Statement::Tag::SUSPEND: {
                auto *su = static_cast<const SuspendStmt *>(stmt);
                for (auto const &fe : su->frame_exports()) {
                    if (!walk_expr(walk_expr, fe.value, d + 1u)) { return false; }
                }
                for (auto const &ext : su->extensions()) {
                    if (ext == nullptr || !ext->binding_projections().empty()) { return false; }
                }
                for (auto *bv : su->extension_binding_values()) {
                    if (!walk_expr(walk_expr, bv, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::PRINT: {
                for (auto *arg : static_cast<const PrintStmt *>(stmt)->arguments()) {
                    if (!walk_expr(walk_expr, arg, d + 1u)) { return false; }
                }
                return true;
            }
            case Statement::Tag::DEBUG_BREAK:
                return false;
        }
        return false;
    };

    if (!walk_stmt(walk_stmt, fb->body(), 0u)) { return false; }

    // Keep EASTL and std swap overloads out of argument-dependent lookup.
    struct CallableRef {
        const FunctionBuilder *builder;
    };
    luisa::vector<CallableRef> remaining;
    for (auto const &callee : fb->custom_callables()) {
        if (callee == nullptr) { return false; }
        if (!visited.contains(callee.get())) { remaining.emplace_back(CallableRef{callee.get()}); }
    }
    luisa::sort(remaining.begin(), remaining.end(), [](CallableRef lhs, CallableRef rhs) noexcept {
        auto *a = lhs.builder;
        auto *b = rhs.builder;
        if (a->hash() != b->hash()) { return a->hash() < b->hash(); }
        if (a->tag() != b->tag()) { return to_underlying(a->tag()) < to_underlying(b->tag()); }
        return a->name() < b->name();
    });
    for (auto [callee] : remaining) {
        if (!collect_and_hash_function_tree(
                callee, false, visited, visiting, hasher, fn_depth + 1u)) {
            return false;
        }
    }

    visiting.erase(fb);
    visited.emplace(fb);
    hasher.add_u64(fb->hash());
    hasher.add_u64(static_cast<uint64_t>(to_underlying(fb->tag())));
    hasher.add_u64(fb->requires_noinline() ? 1u : 0u);
    hasher.add_str(fb->name());
    hasher.add_u64(static_cast<uint64_t>(fb->arguments().size()));
    hasher.add_u64(static_cast<uint64_t>(fb->bound_arguments().size()));
    for (auto const &binding : fb->bound_arguments()) {
        hasher.add_u64(static_cast<uint64_t>(binding.index()));
    }
    return true;
}

[[nodiscard]] bool compute_coro_cache_key(
    const FunctionBuilder &source, uint64_t &out_hi, uint64_t &out_lo) noexcept {
    KeyHasher hasher;
    hasher.add_u64(kCacheFormatVersion);
    hasher.add_u64(static_cast<uint64_t>(LUISA_COMPUTE_VERSION));
    hasher.add_u64(static_cast<uint64_t>(sizeof(size_t)));
    hasher.add_u64(static_cast<uint64_t>(sizeof(void *)));
    hasher.add_u64(0x01020304u);

    luisa::unordered_set<const FunctionBuilder *> visited;
    luisa::unordered_set<const FunctionBuilder *> visiting;
    if (!collect_and_hash_function_tree(&source, true, visited, visiting, hasher)) {
        return false;
    }
    hasher.finalize(out_hi, out_lo);
    return true;
}

class MetaWriter {
public:
    luisa::vector<std::byte> bytes;
    luisa::vector<const Type *> types;
    luisa::unordered_map<const Type *, uint32_t> type_to_id;

    template<typename T>
    void write(T const &v) noexcept {
        static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
        auto old_sz = bytes.size();
        luisa::enlarge_by(bytes, sizeof(T));
        std::memcpy(bytes.data() + old_sz, &v, sizeof(T));
    }

    void write_bytes(const void *ptr, size_t sz) noexcept {
        if (sz == 0u) { return; }
        auto old_sz = bytes.size();
        luisa::enlarge_by(bytes, sz);
        std::memcpy(bytes.data() + old_sz, ptr, sz);
    }

    void write_string(luisa::string_view s) noexcept {
        write<uint32_t>(static_cast<uint32_t>(s.size()));
        write_bytes(s.data(), s.size());
    }

    [[nodiscard]] uint32_t intern_type(const Type *type) noexcept {
        if (type == nullptr) { return 0u; }
        auto [it, inserted] = type_to_id.try_emplace(type, static_cast<uint32_t>(types.size() + 1u));
        if (inserted) {
            if (type->description().empty()) { return UINT32_MAX; }
            types.emplace_back(type);
        }
        return it->second;
    }

    void write_u64_vec(luisa::span<const size_t> vals) noexcept {
        write<uint32_t>(static_cast<uint32_t>(vals.size()));
        for (size_t v : vals) { write<uint64_t>(static_cast<uint64_t>(v)); }
    }

    void write_u32_vec(luisa::span<const uint32_t> vals) noexcept {
        write<uint32_t>(static_cast<uint32_t>(vals.size()));
        for (uint32_t v : vals) { write(v); }
    }
};

class MetaReader {
    const std::byte *_ptr{nullptr};
    const std::byte *_end{nullptr};

public:
    luisa::vector<const Type *> types;

    explicit MetaReader(luisa::span<const std::byte> data) noexcept
        : _ptr{data.data()}, _end{data.data() + data.size()} {}

    [[nodiscard]] size_t remaining() const noexcept {
        return static_cast<size_t>(_end - _ptr);
    }

    template<typename T>
    [[nodiscard]] bool read(T &out) noexcept {
        static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
        if (remaining() < sizeof(T)) { return false; }
        std::memcpy(&out, _ptr, sizeof(T));
        _ptr += sizeof(T);
        return true;
    }

    [[nodiscard]] bool read_string(luisa::string &out) noexcept {
        uint32_t len = 0u;
        if (!read(len) || len > kMaxMetaCount || remaining() < len) { return false; }
        out.resize(len);
        if (len != 0u) {
            std::memcpy(out.data(), _ptr, len);
            _ptr += len;
        }
        return true;
    }

    [[nodiscard]] bool read_u64_vec(luisa::vector<size_t> &out) noexcept {
        uint32_t n = 0u;
        if (!read(n) || n > kMaxMetaCount) { return false; }
        out.resize(n);
        for (uint32_t i = 0u; i < n; ++i) {
            uint64_t v = 0u;
            if (!read(v)) { return false; }
            out[i] = static_cast<size_t>(v);
        }
        return true;
    }

    [[nodiscard]] bool read_u32_vec(luisa::vector<uint32_t> &out) noexcept {
        uint32_t n = 0u;
        if (!read(n) || n > kMaxMetaCount) { return false; }
        out.resize(n);
        for (uint32_t i = 0u; i < n; ++i) {
            if (!read(out[i])) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool resolve_nullable_type(uint32_t tid, const Type *&out) const noexcept {
        if (tid >= types.size()) { return false; }
        out = types[tid];
        return true;
    }

    [[nodiscard]] bool resolve_nonnull_type(uint32_t tid, const Type *&out) const noexcept {
        if (tid == 0u || tid >= types.size() || types[tid] == nullptr) { return false; }
        out = types[tid];
        return true;
    }
};

class LazyCoroXirCacheProvider final : public FunctionBuilder::XIRCacheProvider {
    luisa::filesystem::path _file_path;
    uint64_t _xir_offset{0u};
    uint64_t _xir_size{0u};
    uint64_t _xir_checksum{0u};
    mutable std::once_flag _load_once;
    mutable luisa::shared_ptr<const xir::Module> _module;
    mutable luisa::vector<const xir::FunctionDefinition *> _functions_by_index;

    void _ensure_loaded() const noexcept {
        if (_xir_size == 0u) { return; }
        luisa::vector<std::byte> bytes;
        if (!read_file_bytes(_file_path, bytes,
                             static_cast<size_t>(_xir_offset),
                             static_cast<size_t>(_xir_size))) {
            return;
        }
        auto checksum = luisa::hash64(bytes.data(), bytes.size(), hash64_default_seed);
        if (checksum != _xir_checksum) { return; }
        auto parsed = xir::xir_from_bitcode(bytes);
        if (!parsed.succeeded()) { return; }
        luisa::shared_ptr<xir::Module> shared_mod{std::move(parsed.module)};
        _functions_by_index.reserve(shared_mod->function_list().count_size());
        for (auto *f : shared_mod->function_list()) {
            _functions_by_index.emplace_back(f->definition());
        }
        _module = std::move(shared_mod);
    }

public:
    LazyCoroXirCacheProvider(
        luisa::filesystem::path file_path,
        uint64_t xir_offset,
        uint64_t xir_size,
        uint64_t xir_checksum) noexcept
        : _file_path{std::move(file_path)},
          _xir_offset{xir_offset},
          _xir_size{xir_size},
          _xir_checksum{xir_checksum} {}

    [[nodiscard]] luisa::shared_ptr<const xir::FunctionDefinition>
    load(size_t index) const noexcept override {
        std::call_once(_load_once, [this]() noexcept { _ensure_loaded(); });
        if (!_module || index >= _functions_by_index.size()) { return nullptr; }
        auto *def = _functions_by_index[index];
        if (def == nullptr) { return nullptr; }
        return {_module, def};
    }
};

}// namespace coro_compile_cache_detail

struct CoroCompileCacheCodec {
    using MetaWriter = coro_compile_cache_detail::MetaWriter;
    using MetaReader = coro_compile_cache_detail::MetaReader;

    static void write_slot_set(const coro::CoroGraph::SlotSet &s, MetaWriter &w) noexcept {
        w.write_u64_vec(s.frame_values);
        w.write_u64_vec(s.slots);
    }

    [[nodiscard]] static bool read_slot_set(coro::CoroGraph::SlotSet &s, MetaReader &r) noexcept {
        return r.read_u64_vec(s.frame_values) && r.read_u64_vec(s.slots);
    }

    [[nodiscard]] static bool collect_slot_access_types(
        const coro::CoroSlotAccess &a, MetaWriter &w, size_t d = 0u) noexcept {
        if (d >= coro_compile_cache_detail::kMaxCoroDepth) { return false; }
        if (a._type != nullptr && w.intern_type(a._type) == UINT32_MAX) { return false; }
        for (auto const &p : a._pieces) {
            if (p.logical_type == nullptr || p.physical_type == nullptr ||
                w.intern_type(p.logical_type) == UINT32_MAX ||
                w.intern_type(p.physical_type) == UINT32_MAX) {
                return false;
            }
        }
        for (auto const &alt : a._alternatives) {
            if (!collect_slot_access_types(alt, w, d + 1u)) { return false; }
        }
        for (auto const &cond : a._conditions) {
            if (!collect_slot_access_types(cond, w, d + 1u)) { return false; }
        }
        return true;
    }

    [[nodiscard]] static bool write_slot_access(
        const coro::CoroSlotAccess &a, MetaWriter &w, size_t d = 0u) noexcept {
        if (d >= coro_compile_cache_detail::kMaxCoroDepth) { return false; }
        w.write<uint32_t>(w.intern_type(a._type));
        w.write(a._access);
        w.write(a._lifetime);
        w.write<uint32_t>(static_cast<uint32_t>(a._pieces.size()));
        for (auto const &p : a._pieces) {
            w.write<uint64_t>(static_cast<uint64_t>(p.frame_value_index));
            w.write<uint64_t>(static_cast<uint64_t>(p.field_index));
            w.write_u32_vec(p.access_chain);
            w.write<uint32_t>(w.intern_type(p.logical_type));
            w.write<uint32_t>(w.intern_type(p.physical_type));
            w.write<uint8_t>(p.bit_offset.has_value() ? 1u : 0u);
            if (p.bit_offset) { w.write(*p.bit_offset); }
        }
        w.write<uint32_t>(static_cast<uint32_t>(a._alternatives.size()));
        for (auto const &alt : a._alternatives) {
            if (!write_slot_access(alt, w, d + 1u)) { return false; }
        }
        w.write<uint32_t>(static_cast<uint32_t>(a._conditions.size()));
        for (auto const &cond : a._conditions) {
            if (!write_slot_access(cond, w, d + 1u)) { return false; }
        }
        w.write_u64_vec(a._use_frame_values);
        w.write_u64_vec(a._def_frame_values);
        w.write_u64_vec(a._use_slots);
        w.write_u64_vec(a._def_slots);
        w.write_u64_vec(a._rmw_slots);
        w.write_u64_vec(a._reconstruct_slots);
        return true;
    }

    [[nodiscard]] static bool read_slot_access(
        coro::CoroSlotAccess &a, MetaReader &r, size_t d = 0u) noexcept {
        if (d >= coro_compile_cache_detail::kMaxCoroDepth) { return false; }
        uint32_t tid = 0u;
        if (!r.read(tid) || !r.resolve_nullable_type(tid, a._type) ||
            !r.read(a._access) || !r.read(a._lifetime)) {
            return false;
        }
        uint32_t piece_count = 0u;
        if (!r.read(piece_count) || piece_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        a._pieces.resize(piece_count);
        for (uint32_t i = 0u; i < piece_count; ++i) {
            auto &p = a._pieces[i];
            uint64_t fvi = 0u, fi = 0u;
            uint32_t ltid = 0u, ptid = 0u;
            uint8_t has_bit = 0u;
            if (!r.read(fvi) || !r.read(fi) || !r.read_u32_vec(p.access_chain) ||
                !r.read(ltid) || !r.resolve_nonnull_type(ltid, p.logical_type) ||
                !r.read(ptid) || !r.resolve_nonnull_type(ptid, p.physical_type) ||
                !r.read(has_bit)) {
                return false;
            }
            p.frame_value_index = static_cast<size_t>(fvi);
            p.field_index = static_cast<size_t>(fi);
            if (has_bit != 0u) {
                uint32_t bo = 0u;
                if (!r.read(bo)) { return false; }
                p.bit_offset = bo;
            }
        }
        uint32_t alt_count = 0u;
        if (!r.read(alt_count) || alt_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        a._alternatives.resize(alt_count);
        for (uint32_t i = 0u; i < alt_count; ++i) {
            if (!read_slot_access(a._alternatives[i], r, d + 1u)) { return false; }
        }
        uint32_t cond_count = 0u;
        if (!r.read(cond_count) || cond_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        a._conditions.resize(cond_count);
        for (uint32_t i = 0u; i < cond_count; ++i) {
            if (!read_slot_access(a._conditions[i], r, d + 1u)) { return false; }
        }
        return r.read_u64_vec(a._use_frame_values) &&
               r.read_u64_vec(a._def_frame_values) &&
               r.read_u64_vec(a._use_slots) &&
               r.read_u64_vec(a._def_slots) &&
               r.read_u64_vec(a._rmw_slots) &&
               r.read_u64_vec(a._reconstruct_slots);
    }

    [[nodiscard]] static bool encode_meta(
        const CoroutineCompileResult &result,
        luisa::span<const size_t> xir_func_indices,
        luisa::vector<std::byte> &out_meta) noexcept {
        MetaWriter body;

        // Intern all types used by frame_desc & boundary bindings
        for (auto const &f : result.frame_desc._fields) {
            if (f.type == nullptr || body.intern_type(f.type) == UINT32_MAX) { return false; }
        }
        for (auto const &b : result.graph._boundaries) {
            for (auto const &acc : b.bindings) {
                if (!collect_slot_access_types(acc, body)) { return false; }
            }
        }

        body.write<uint64_t>(static_cast<uint64_t>(result.boundary_verifier_count));
        body.write<uint64_t>(static_cast<uint64_t>(result.nested_pass_boundary_verifier_count));
        body.write_u32_vec(result.trigger_tokens);
        body.write<uint32_t>(static_cast<uint32_t>(result.subroutine_source_argument_indices.size()));
        for (auto const &indices : result.subroutine_source_argument_indices) {
            body.write_u64_vec(indices);
        }
        body.write<uint32_t>(static_cast<uint32_t>(xir_func_indices.size()));
        for (size_t idx : xir_func_indices) {
            body.write<uint64_t>(static_cast<uint64_t>(idx));
        }

        // CoroFrameDesc
        body.write<uint64_t>(static_cast<uint64_t>(result.frame_desc._total_size));
        body.write<uint32_t>(static_cast<uint32_t>(result.frame_desc._fields.size()));
        for (auto const &f : result.frame_desc._fields) {
            body.write_string(f.name);
            body.write<uint32_t>(body.intern_type(f.type));
        }
        luisa::vector<std::pair<luisa::string_view, size_t>> extra_aliases;
        for (auto const &[name, idx] : result.frame_desc._field_indices) {
            if (idx >= result.frame_desc._fields.size()) { return false; }
            if (result.frame_desc._fields[idx].name != name) {
                extra_aliases.emplace_back(luisa::string_view{name}, idx);
            }
        }
        luisa::sort(extra_aliases.begin(), extra_aliases.end(), [](auto const &a, auto const &b) noexcept {
            if (a.second != b.second) { return a.second < b.second; }
            return a.first < b.first;
        });
        body.write<uint32_t>(static_cast<uint32_t>(extra_aliases.size()));
        for (auto const &[name, idx] : extra_aliases) {
            body.write_string(name);
            body.write<uint64_t>(static_cast<uint64_t>(idx));
        }

        // CoroGraph._call_graph
        auto const &cg = result.graph._call_graph;
        body.write<uint64_t>(static_cast<uint64_t>(cg.analysis_state_count));
        body.write<uint32_t>(static_cast<uint32_t>(cg.functions.size()));
        for (auto const &fn : cg.functions) {
            body.write(fn.id);
            body.write_u32_vec(fn.resume_tokens);
        }
        body.write<uint32_t>(static_cast<uint32_t>(cg.edges.size()));
        for (auto const &edge : cg.edges) {
            body.write(edge.caller);
            body.write(edge.callee);
            body.write(edge.return_site);
        }

        // CoroGraph._nodes
        body.write<uint32_t>(static_cast<uint32_t>(result.graph._nodes.size()));
        for (auto const &node : result.graph._nodes) {
            body.write<uint64_t>(static_cast<uint64_t>(node.index));
            body.write_string(node.name);
            body.write<uint64_t>(static_cast<uint64_t>(node.token));
            body.write<uint8_t>(node.is_terminal ? 1u : 0u);
            body.write_u64_vec(node.input_fields);
            body.write_u64_vec(node.output_fields);
            body.write_u64_vec(node.relocation_fields);
            body.write_u64_vec(node.targets);
        }

        // CoroGraph._edges
        body.write<uint32_t>(static_cast<uint32_t>(result.graph._edges.size()));
        for (auto const &edge : result.graph._edges) {
            body.write<uint64_t>(static_cast<uint64_t>(edge.from_index));
            body.write<uint64_t>(static_cast<uint64_t>(edge.to_index));
            body.write_u64_vec(edge.load_fields);
            body.write_u64_vec(edge.store_fields);
        }

        // CoroGraph._boundaries
        body.write<uint32_t>(static_cast<uint32_t>(result.graph._boundaries.size()));
        for (auto const &b : result.graph._boundaries) {
            body.write<uint64_t>(static_cast<uint64_t>(b.index));
            body.write<uint64_t>(static_cast<uint64_t>(b.from_index));
            body.write<uint64_t>(static_cast<uint64_t>(b.to_index));
            body.write<uint64_t>(static_cast<uint64_t>(b.token));
            body.write<uint32_t>(static_cast<uint32_t>(b.extensions.size()));
            for (auto const &ext : b.extensions) {
                if (ext == nullptr) {
                    body.write<uint8_t>(0u);
                    continue;
                }
                if (!ext->binding_projections().empty()) { return false; }
                body.write<uint8_t>(1u);
                body.write_string(ext->schema());
                body.write<uint32_t>(ext->version());
                body.write<uint8_t>(ext->is_annotation() ? 1u : 0u);
                body.write(ext->fallback());
                body.write<uint32_t>(static_cast<uint32_t>(ext->bindings().size()));
                for (auto const &eb : ext->bindings()) {
                    body.write_string(eb.name);
                    body.write(eb.access);
                    body.write(eb.lifetime);
                    body.write(eb.index);
                }
                body.write<uint32_t>(static_cast<uint32_t>(ext->attributes().size()));
                for (auto const &attr : ext->attributes()) {
                    body.write_string(attr.name);
                    body.write<uint8_t>(static_cast<uint8_t>(attr.value.index()));
                    luisa::visit([&]<typename V>(V const &val) noexcept {
                        if constexpr (std::is_same_v<V, luisa::string>) {
                            body.write_string(val);
                        } else {
                            body.write(val);
                        }
                    }, attr.value);
                }
            }
            body.write<uint32_t>(static_cast<uint32_t>(b.bindings.size()));
            for (auto const &acc : b.bindings) {
                if (!write_slot_access(acc, body)) { return false; }
            }
            write_slot_set(b.source_store, body);
            write_slot_set(b.target_live, body);
            body.write<uint32_t>(static_cast<uint32_t>(b.stages.size()));
            for (auto const &st : b.stages) {
                body.write<uint64_t>(static_cast<uint64_t>(st.extension_index));
                write_slot_set(st.use, body);
                write_slot_set(st.def, body);
                write_slot_set(st.live_in, body);
                write_slot_set(st.live_out, body);
                write_slot_set(st.preserve, body);
                write_slot_set(st.required_def, body);
                body.write_u64_vec(st.rmw_slots);
                body.write_u64_vec(st.reconstruct_slots);
                body.write(st.execution_hint);
                body.write_string(st.async_domain);
            }
        }

        // Prepend Type Table before meta body
        MetaWriter full;
        full.write<uint32_t>(static_cast<uint32_t>(body.types.size()));
        for (auto *t : body.types) {
            full.write_string(t->description());
        }
        full.write_bytes(body.bytes.data(), body.bytes.size());
        out_meta = std::move(full.bytes);
        return true;
    }

    [[nodiscard]] static bool decode_meta(
        luisa::span<const std::byte> meta_bytes,
        CoroutineCompileResult &out_result,
        luisa::vector<size_t> &out_xir_func_indices) noexcept {
        MetaReader r{meta_bytes};
        uint32_t type_count = 0u;
        if (!r.read(type_count) || type_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        r.types.reserve(static_cast<size_t>(type_count) + 1u);
        r.types.emplace_back(nullptr);
        for (uint32_t i = 0u; i < type_count; ++i) {
            luisa::string desc;
            if (!r.read_string(desc) || desc.empty() || desc == "void") { return false; }
            auto *t = Type::from(desc);
            if (t == nullptr) { return false; }
            r.types.emplace_back(t);
        }

        uint64_t bvc = 0u, nbvc = 0u;
        if (!r.read(bvc) || !r.read(nbvc) ||
            !r.read_u32_vec(out_result.trigger_tokens)) {
            return false;
        }
        out_result.boundary_verifier_count = static_cast<size_t>(bvc);
        out_result.nested_pass_boundary_verifier_count = static_cast<size_t>(nbvc);

        uint32_t src_arg_count = 0u;
        if (!r.read(src_arg_count) || src_arg_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        out_result.subroutine_source_argument_indices.resize(src_arg_count);
        for (uint32_t i = 0u; i < src_arg_count; ++i) {
            if (!r.read_u64_vec(out_result.subroutine_source_argument_indices[i])) { return false; }
        }
        if (!r.read_u64_vec(out_xir_func_indices)) { return false; }

        // CoroFrameDesc
        uint64_t expected_total_size = 0u;
        uint32_t field_count = 0u;
        if (!r.read(expected_total_size) || !r.read(field_count) ||
            field_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        out_result.frame_desc._reset();
        out_result.frame_desc._fields.reserve(field_count);
        for (uint32_t i = 0u; i < field_count; ++i) {
            luisa::string name;
            uint32_t tid = 0u;
            const Type *ft = nullptr;
            if (!r.read_string(name) || name.empty() ||
                !r.read(tid) || !r.resolve_nonnull_type(tid, ft) ||
                out_result.frame_desc._field_indices.contains(name)) {
                return false;
            }
            out_result.frame_desc.add_field(std::move(name), ft);
        }
        if (out_result.frame_desc._total_size != static_cast<size_t>(expected_total_size)) {
            return false;
        }
        uint32_t alias_count = 0u;
        if (!r.read(alias_count) || alias_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        for (uint32_t i = 0u; i < alias_count; ++i) {
            luisa::string name;
            uint64_t idx = 0u;
            if (!r.read_string(name) || name.empty() || !r.read(idx) ||
                !out_result.frame_desc._add_alias(name, static_cast<size_t>(idx))) {
                return false;
            }
        }

        // CoroGraph._call_graph
        auto &cg = out_result.graph._call_graph;
        uint64_t asc = 0u;
        uint32_t cg_fn_count = 0u, cg_edge_count = 0u;
        if (!r.read(asc) || !r.read(cg_fn_count) ||
            cg_fn_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        cg.analysis_state_count = static_cast<size_t>(asc);
        cg.functions.resize(cg_fn_count);
        for (uint32_t i = 0u; i < cg_fn_count; ++i) {
            if (!r.read(cg.functions[i].id) ||
                !r.read_u32_vec(cg.functions[i].resume_tokens)) {
                return false;
            }
        }
        if (!r.read(cg_edge_count) || cg_edge_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        cg.edges.resize(cg_edge_count);
        for (uint32_t i = 0u; i < cg_edge_count; ++i) {
            if (!r.read(cg.edges[i].caller) || !r.read(cg.edges[i].callee) ||
                !r.read(cg.edges[i].return_site)) {
                return false;
            }
        }

        // CoroGraph._nodes
        uint32_t node_count = 0u;
        if (!r.read(node_count) || node_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        out_result.graph._nodes.reserve(node_count);
        out_result.graph._token_to_index.clear();
        out_result.graph._name_to_index.clear();
        for (uint32_t i = 0u; i < node_count; ++i) {
            coro::CoroGraph::Node node;
            uint64_t idx = 0u, tok = 0u;
            uint8_t is_term = 0u;
            if (!r.read(idx) || static_cast<size_t>(idx) != i ||
                !r.read_string(node.name) || !r.read(tok) || !r.read(is_term) ||
                !r.read_u64_vec(node.input_fields) ||
                !r.read_u64_vec(node.output_fields) ||
                !r.read_u64_vec(node.relocation_fields) ||
                !r.read_u64_vec(node.targets)) {
                return false;
            }
            node.index = i;
            node.token = static_cast<size_t>(tok);
            node.is_terminal = is_term != 0u;
            node.callable = nullptr;
            out_result.graph._nodes.push_back(std::move(node));
            auto &stored = out_result.graph._nodes.back();
            if (!out_result.graph._token_to_index.emplace(stored.token, i).second) {
                return false;
            }
            if (!stored.name.empty()) {
                out_result.graph._name_to_index.emplace(stored.name, i);
            }
        }

        // CoroGraph._edges
        uint32_t edge_count = 0u;
        if (!r.read(edge_count) || edge_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        out_result.graph._edges.resize(edge_count);
        for (uint32_t i = 0u; i < edge_count; ++i) {
            uint64_t fi = 0u, ti = 0u;
            if (!r.read(fi) || !r.read(ti) ||
                !r.read_u64_vec(out_result.graph._edges[i].load_fields) ||
                !r.read_u64_vec(out_result.graph._edges[i].store_fields)) {
                return false;
            }
            out_result.graph._edges[i].from_index = static_cast<size_t>(fi);
            out_result.graph._edges[i].to_index = static_cast<size_t>(ti);
        }

        // CoroGraph._boundaries
        uint32_t boundary_count = 0u;
        if (!r.read(boundary_count) || boundary_count > coro_compile_cache_detail::kMaxMetaCount) {
            return false;
        }
        out_result.graph._boundaries.resize(boundary_count);
        for (uint32_t bi = 0u; bi < boundary_count; ++bi) {
            auto &b = out_result.graph._boundaries[bi];
            uint64_t idx = 0u, fi = 0u, ti = 0u, tok = 0u;
            uint32_t ext_count = 0u;
            if (!r.read(idx) || !r.read(fi) || !r.read(ti) || !r.read(tok) ||
                !r.read(ext_count) || ext_count > coro_compile_cache_detail::kMaxMetaCount) {
                return false;
            }
            b.index = static_cast<size_t>(idx);
            b.from_index = static_cast<size_t>(fi);
            b.to_index = static_cast<size_t>(ti);
            b.token = static_cast<size_t>(tok);
            b.extensions.reserve(ext_count);
            for (uint32_t ei = 0u; ei < ext_count; ++ei) {
                uint8_t present = 0u;
                if (!r.read(present)) { return false; }
                if (present == 0u) {
                    b.extensions.emplace_back(nullptr);
                    continue;
                }
                luisa::string schema;
                uint32_t version = 0u;
                uint8_t is_annotation = 0u;
                CoroSuspendFallback fallback{};
                uint32_t b_cnt = 0u, a_cnt = 0u;
                if (!r.read_string(schema) || schema.empty() ||
                    !r.read(version) || version == 0u ||
                    !r.read(is_annotation) || !r.read(fallback) ||
                    !r.read(b_cnt) || b_cnt > coro_compile_cache_detail::kMaxMetaCount) {
                    return false;
                }
                luisa::vector<CoroSuspendBinding> bindings;
                bindings.reserve(b_cnt);
                for (uint32_t j = 0u; j < b_cnt; ++j) {
                    CoroSuspendBinding sb;
                    if (!r.read_string(sb.name) || sb.name.empty() ||
                        !r.read(sb.access) || !r.read(sb.lifetime) || !r.read(sb.index)) {
                        return false;
                    }
                    for (uint32_t k = 0u; k < j; ++k) {
                        if (bindings[k].name == sb.name) { return false; }
                    }
                    bindings.emplace_back(std::move(sb));
                }
                if (!r.read(a_cnt) || a_cnt > coro_compile_cache_detail::kMaxMetaCount) {
                    return false;
                }
                luisa::vector<CoroSuspendAttribute> attributes;
                attributes.reserve(a_cnt);
                for (uint32_t j = 0u; j < a_cnt; ++j) {
                    CoroSuspendAttribute attr;
                    uint8_t vtag = 0u;
                    if (!r.read_string(attr.name) || attr.name.empty() || !r.read(vtag)) {
                        return false;
                    }
                    switch (vtag) {
                        case 0u: {
                            bool v{};
                            if (!r.read(v)) { return false; }
                            attr.value = v;
                            break;
                        }
                        case 1u: {
                            int64_t v{};
                            if (!r.read(v)) { return false; }
                            attr.value = v;
                            break;
                        }
                        case 2u: {
                            uint64_t v{};
                            if (!r.read(v)) { return false; }
                            attr.value = v;
                            break;
                        }
                        case 3u: {
                            double v{};
                            if (!r.read(v)) { return false; }
                            attr.value = v;
                            break;
                        }
                        case 4u: {
                            luisa::string v;
                            if (!r.read_string(v)) { return false; }
                            attr.value = std::move(v);
                            break;
                        }
                        default:
                            return false;
                    }
                    for (uint32_t k = 0u; k < j; ++k) {
                        if (attributes[k].name == attr.name) { return false; }
                    }
                    attributes.emplace_back(std::move(attr));
                }
                b.extensions.emplace_back(
                    is_annotation != 0u ?
                        make_coro_suspend_annotation_data(
                            std::move(schema), version, fallback,
                            std::move(bindings), std::move(attributes)) :
                        make_coro_suspend_extension_data(
                            std::move(schema), version, fallback,
                            std::move(bindings), std::move(attributes)));
            }

            uint32_t bind_count = 0u;
            if (!r.read(bind_count) || bind_count > coro_compile_cache_detail::kMaxMetaCount) {
                return false;
            }
            b.bindings.resize(bind_count);
            for (uint32_t j = 0u; j < bind_count; ++j) {
                if (!read_slot_access(b.bindings[j], r)) { return false; }
            }
            if (!read_slot_set(b.source_store, r) || !read_slot_set(b.target_live, r)) {
                return false;
            }
            uint32_t stage_count = 0u;
            if (!r.read(stage_count) || stage_count > coro_compile_cache_detail::kMaxMetaCount) {
                return false;
            }
            b.stages.resize(stage_count);
            for (uint32_t si = 0u; si < stage_count; ++si) {
                auto &st = b.stages[si];
                uint64_t ext_idx = 0u;
                if (!r.read(ext_idx) ||
                    !read_slot_set(st.use, r) ||
                    !read_slot_set(st.def, r) ||
                    !read_slot_set(st.live_in, r) ||
                    !read_slot_set(st.live_out, r) ||
                    !read_slot_set(st.preserve, r) ||
                    !read_slot_set(st.required_def, r) ||
                    !r.read_u64_vec(st.rmw_slots) ||
                    !r.read_u64_vec(st.reconstruct_slots) ||
                    !r.read(st.execution_hint) ||
                    !r.read_string(st.async_domain)) {
                    return false;
                }
                st.extension_index = static_cast<size_t>(ext_idx);
            }
        }
        return r.remaining() == 0u;
    }
};

CoroCompileCacheLookup coro_compile_cache_try_load(
    const FunctionBuilder &source_builder,
    CoroutineCompileResult &out_result) noexcept {

    CoroCompileCacheLookup lookup{};
    if (!coro_compile_cache_detail::cache_enabled() ||
        coro_compile_cache_detail::any_diagnostic_flag_enabled()) {
        return lookup;
    }
    if (!coro_compile_cache_detail::compute_coro_cache_key(
            source_builder, lookup.key_hi, lookup.key_lo)) {
        return lookup;
    }
    lookup.store_allowed = true;

    const auto cache_dir = coro_compile_cache_detail::resolve_cache_dir();
    if (cache_dir.empty()) {
        lookup.store_allowed = false;
        return lookup;
    }
    const auto cache_path = coro_compile_cache_detail::make_cache_file_path(
        cache_dir, lookup.key_hi, lookup.key_lo);

    std::error_code ec;
    auto file_size_u64 = luisa::filesystem::file_size(cache_path, ec);
    if (ec || file_size_u64 < sizeof(coro_compile_cache_detail::CoroCacheFileHeader)) {
        return lookup;
    }

    // Read only header + meta + ast on cache hit (leave optional XIR tail unread).
    luisa::vector<std::byte> header_bytes;
    if (!coro_compile_cache_detail::read_file_bytes(
            cache_path, header_bytes, 0u,
            sizeof(coro_compile_cache_detail::CoroCacheFileHeader))) {
        return lookup;
    }
    coro_compile_cache_detail::CoroCacheFileHeader hdr{};
    std::memcpy(&hdr, header_bytes.data(), sizeof(hdr));
    if (hdr.magic != coro_compile_cache_detail::kCacheMagic ||
        hdr.format_version != coro_compile_cache_detail::kCacheFormatVersion ||
        hdr.key_hi != lookup.key_hi ||
        hdr.key_lo != lookup.key_lo) {
        return lookup;
    }
    auto check_range = [&](uint64_t off, uint64_t sz) noexcept {
        return off >= sizeof(hdr) && off <= file_size_u64 && sz <= (file_size_u64 - off);
    };
    if (!check_range(hdr.meta_offset, hdr.meta_size) ||
        !check_range(hdr.ast_offset, hdr.ast_size) ||
        (hdr.xir_size != 0u && !check_range(hdr.xir_offset, hdr.xir_size))) {
        return lookup;
    }

    luisa::vector<std::byte> meta_bytes;
    if (!coro_compile_cache_detail::read_file_bytes(
            cache_path, meta_bytes,
            static_cast<size_t>(hdr.meta_offset),
            static_cast<size_t>(hdr.meta_size))) {
        return lookup;
    }
    if (luisa::hash64(meta_bytes.data(), meta_bytes.size(), hash64_default_seed) != hdr.meta_checksum) {
        return lookup;
    }

    luisa::vector<std::byte> ast_bytes;
    if (!coro_compile_cache_detail::read_file_bytes(
            cache_path, ast_bytes,
            static_cast<size_t>(hdr.ast_offset),
            static_cast<size_t>(hdr.ast_size))) {
        return lookup;
    }
    if (luisa::hash64(ast_bytes.data(), ast_bytes.size(), hash64_default_seed) != hdr.ast_checksum) {
        return lookup;
    }

    CoroutineCompileResult candidate{};
    luisa::vector<size_t> xir_func_indices;
    if (!CoroCompileCacheCodec::decode_meta(meta_bytes, candidate, xir_func_indices)) {
        return lookup;
    }

    luisa::shared_ptr<const FunctionBuilder::XIRCacheProvider> xir_provider;
    if (hdr.xir_size != 0u) {
        xir_provider = luisa::make_shared<coro_compile_cache_detail::LazyCoroXirCacheProvider>(
            cache_path, hdr.xir_offset, hdr.xir_size, hdr.xir_checksum);
    }

    if (!CallableLibrary::deserialize_function_graph(
            ast_bytes, candidate.subroutines,
            std::move(xir_provider), xir_func_indices)) {
        return lookup;
    }

    if (candidate.trigger_tokens.empty() ||
        candidate.trigger_tokens.front() != 0u ||
        candidate.subroutines.size() != candidate.trigger_tokens.size() ||
        candidate.subroutine_source_argument_indices.size() != candidate.subroutines.size() ||
        candidate.graph.node_count() != candidate.subroutines.size()) {
        return lookup;
    }

    out_result = std::move(candidate);
    lookup.hit = true;
    return lookup;
}

void coro_compile_cache_store(
    uint64_t key_hi,
    uint64_t key_lo,
    const CoroutineCompileResult &result,
    const xir::Module *xir_module,
    luisa::span<const xir::FunctionDefinition *const> continuation_definitions) noexcept {

    const auto cache_dir = coro_compile_cache_detail::resolve_cache_dir();
    if (cache_dir.empty()) { return; }

    std::error_code ec;
    luisa::filesystem::create_directories(cache_dir, ec);
    if (ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create coroutine cache directory '{}': {}.",
            luisa::platform::path_to_narrow(cache_dir), ec.message());
        return;
    }

    luisa::vector<std::byte> ast_bytes =
        CallableLibrary::serialize_function_graph(
            luisa::span<const luisa::shared_ptr<const FunctionBuilder>>{
                result.subroutines.data(), result.subroutines.size()});
    if (ast_bytes.empty()) { return; }

    luisa::vector<size_t> xir_func_indices(
        result.subroutines.size(), static_cast<size_t>(-1));
    luisa::vector<std::byte> xir_bytes;
    const auto max_insts = coro_compile_cache_detail::xir_cache_max_insts();
    if (max_insts > 0u && xir_module != nullptr &&
        continuation_definitions.size() == result.subroutines.size()) {
        size_t total_insts = 0u;
        luisa::unordered_map<const xir::FunctionDefinition *, size_t> def_to_idx;
        size_t fn_idx = 0u;
        for (auto *f : xir_module->function_list()) {
            if (auto *def = f->definition()) {
                def_to_idx.emplace(def, fn_idx);
                for (auto *bb : def->basic_blocks()) {
                    total_insts += bb->instructions().count_size();
                }
            }
            ++fn_idx;
        }
        if (total_insts <= max_insts) {
            auto bitcode = xir::xir_to_bitcode(xir_module);
            if (bitcode.succeeded() && !bitcode.bitcode.empty()) {
                xir_bytes = std::move(bitcode.bitcode);
                for (size_t i = 0u; i < continuation_definitions.size(); ++i) {
                    if (result.subroutines[i]->xir_cache_hash() != 0u) {
                        if (auto it = def_to_idx.find(continuation_definitions[i]);
                            it != def_to_idx.end()) {
                            xir_func_indices[i] = it->second;
                        }
                    }
                }
            }
        }
    }

    luisa::vector<std::byte> meta_bytes;
    if (!CoroCompileCacheCodec::encode_meta(result, xir_func_indices, meta_bytes)) {
        return;
    }

    coro_compile_cache_detail::CoroCacheFileHeader hdr{};
    hdr.magic = coro_compile_cache_detail::kCacheMagic;
    hdr.format_version = coro_compile_cache_detail::kCacheFormatVersion;
    hdr.flags = 0u;
    hdr.key_hi = key_hi;
    hdr.key_lo = key_lo;
    hdr.meta_offset = sizeof(hdr);
    hdr.meta_size = meta_bytes.size();
    hdr.meta_checksum = luisa::hash64(meta_bytes.data(), meta_bytes.size(), hash64_default_seed);
    hdr.ast_offset = hdr.meta_offset + hdr.meta_size;
    hdr.ast_size = ast_bytes.size();
    hdr.ast_checksum = luisa::hash64(ast_bytes.data(), ast_bytes.size(), hash64_default_seed);
    if (!xir_bytes.empty()) {
        hdr.xir_offset = hdr.ast_offset + hdr.ast_size;
        hdr.xir_size = xir_bytes.size();
        hdr.xir_checksum = luisa::hash64(xir_bytes.data(), xir_bytes.size(), hash64_default_seed);
    }

    static std::atomic<uint64_t> tmp_seq{0u};
    const auto final_path = coro_compile_cache_detail::make_cache_file_path(
        cache_dir, key_hi, key_lo);
    const auto seq = tmp_seq.fetch_add(1u, std::memory_order_relaxed);
    const auto tmp_name = luisa::format(
        "{}.{}.{}.tmp",
        luisa::platform::path_to_narrow(final_path.filename()),
        coro_compile_cache_detail::current_process_id(),
        seq);
    const auto tmp_path = cache_dir / luisa::platform::path_from_narrow(tmp_name);

    {
        std::ofstream out{tmp_path, std::ios::binary | std::ios::trunc};
        if (!out.is_open()) { return; }
        out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
        if (!meta_bytes.empty()) {
            out.write(reinterpret_cast<const char *>(meta_bytes.data()),
                      static_cast<std::streamsize>(meta_bytes.size()));
        }
        if (!ast_bytes.empty()) {
            out.write(reinterpret_cast<const char *>(ast_bytes.data()),
                      static_cast<std::streamsize>(ast_bytes.size()));
        }
        if (!xir_bytes.empty()) {
            out.write(reinterpret_cast<const char *>(xir_bytes.data()),
                      static_cast<std::streamsize>(xir_bytes.size()));
        }
        out.flush();
        if (!out.good()) {
            out.close();
            luisa::filesystem::remove(tmp_path, ec);
            return;
        }
    }

    luisa::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        luisa::filesystem::remove(tmp_path, ec);
    }
}

}// namespace luisa::compute::detail
