#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/osl/instruction.h>
#include <luisa/osl/literal.h>
#include <luisa/osl/shader.h>
#include <luisa/osl/shader_group.h>
#include <luisa/osl/string_table.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/translator.h>
#include <luisa/osl/type.h>

namespace luisa::compute::osl::detail {

// --- Abstract OSO CFG representation shared by analyses and XIR lowering ---

enum struct OSOTermKind : uint32_t {
    BR,
    COND_BR,
    UNREACHABLE,
    EXIT_LAYER,
};

struct OSOBasicBlock {
    uint32_t id{0u};
    luisa::vector<uint32_t> pcs;
    OSOTermKind term{OSOTermKind::BR};
    uint32_t cond_pc{~0u};
    const Symbol *cond_symbol{nullptr};
    uint32_t succ[2]{~0u, ~0u};// [0]=target/true, [1]=false
};

struct OSOCodeRange {
    luisa::string identifier;
    uint32_t begin_pc{0u};
    uint32_t end_pc{0u};
    bool is_main{false};
    const Symbol *param_symbol{nullptr};// root symbol for code <param>
    luisa::vector<OSOBasicBlock> blocks;
    uint32_t entry_block{0u};
    uint32_t exit_block{0u};
};

// --- Linked ShaderGroup representation ---

struct LinkedConnection {
    uint32_t src_layer{0u};
    const Symbol *src_symbol{nullptr};
    uint32_t dst_layer{0u};
    const Symbol *dst_symbol{nullptr};
};

struct LinkedOutput {
    uint32_t layer_index{0u};
    const Symbol *symbol{nullptr};
    bool want_deriv{false};
};

struct LinkedLayer {
    uint32_t group_layer_index{0u};
    uint32_t active_index{0u};
    const ShaderLayer *layer{nullptr};
    const Shader *shader{nullptr};
    luisa::vector<uint32_t> incoming_connections;
    luisa::unordered_map<const Symbol *, int> effective_array_length;
    luisa::unordered_map<const Symbol *, const ParameterValue *> param_overrides;
    luisa::unordered_map<const Symbol *, bool> dynamic_inputs;
    luisa::unordered_set<const Symbol *> skip_init_params;
    luisa::vector<OSOCodeRange> ranges;
};

struct LinkedGroup {
    luisa::vector<LinkedLayer> layers;
    luisa::vector<LinkedConnection> connections;
    luisa::vector<LinkedOutput> outputs;
    size_t eliminated_layer_count{0u};
};

[[nodiscard]] bool link_shader_group(
    const ShaderGroup &group,
    bool prune_dead_layers,
    LinkedGroup &out_group,
    luisa::vector<TranslationError> &errors) noexcept;

// --- String & scalar constant propagation ---

struct StringLattice {
    enum struct Kind : uint8_t {
        BOT,
        SET,
        TOP,
    };
    Kind kind{Kind::BOT};
    luisa::vector<StringToken> tokens;// sorted, unique when kind == SET

    [[nodiscard]] static StringLattice bot() noexcept { return {}; }
    [[nodiscard]] static StringLattice top() noexcept { return {.kind = Kind::TOP}; }
    [[nodiscard]] static StringLattice singleton(StringToken tok) noexcept {
        return {.kind = Kind::SET, .tokens = {tok}};
    }
    [[nodiscard]] bool is_bot() const noexcept { return kind == Kind::BOT; }
    [[nodiscard]] bool is_set() const noexcept { return kind == Kind::SET; }
    [[nodiscard]] bool is_top() const noexcept { return kind == Kind::TOP; }
    [[nodiscard]] bool is_singleton() const noexcept {
        return kind == Kind::SET && tokens.size() == 1u;
    }
    [[nodiscard]] luisa::optional<StringToken> as_singleton() const noexcept {
        return is_singleton() ? luisa::optional<StringToken>{tokens.front()} : luisa::nullopt;
    }
    [[nodiscard]] bool operator==(const StringLattice &rhs) const noexcept {
        return kind == rhs.kind && tokens == rhs.tokens;
    }
    bool join(const StringLattice &other, uint32_t max_size) noexcept;
};

struct ScalarLattice {
    enum struct Kind : uint8_t {
        BOT,
        INT,
        FLOAT,
        TOP,
    };
    Kind kind{Kind::BOT};
    int32_t i{0};
    float f{0.0f};

    [[nodiscard]] static ScalarLattice bot() noexcept { return {}; }
    [[nodiscard]] static ScalarLattice top() noexcept { return {.kind = Kind::TOP}; }
    [[nodiscard]] static ScalarLattice make_int(int32_t v) noexcept {
        return {.kind = Kind::INT, .i = v};
    }
    [[nodiscard]] static ScalarLattice make_float(float v) noexcept {
        return {.kind = Kind::FLOAT, .f = v};
    }
    [[nodiscard]] bool is_bot() const noexcept { return kind == Kind::BOT; }
    [[nodiscard]] bool is_int() const noexcept { return kind == Kind::INT; }
    [[nodiscard]] bool is_float() const noexcept { return kind == Kind::FLOAT; }
    [[nodiscard]] bool is_top() const noexcept { return kind == Kind::TOP; }
    [[nodiscard]] bool operator==(const ScalarLattice &rhs) const noexcept;
    bool join(const ScalarLattice &other) noexcept;
};

struct InstructionAnalysis {
    bool reachable{false};
    luisa::vector<StringLattice> arg_strings;
    luisa::vector<ScalarLattice> arg_scalars;
    bool branch_mixed{false};
    luisa::optional<bool> folded_branch;// for conditional branch pcs
    ScalarLattice folded_result;        // for folding scalar ops if needed
};

struct LayerAnalysis {
    luisa::vector<InstructionAnalysis> instructions;
    luisa::unordered_set<const Symbol *> dynamic_string_symbols;
    luisa::unordered_map<const Symbol *, StringLattice> output_strings;
    luisa::unordered_map<const Symbol *, ScalarLattice> output_scalars;
};

struct GroupStringAnalysis {
    luisa::vector<LayerAnalysis> layers;
};

void analyze_group_strings_and_constants(
    const LinkedGroup &group,
    StringTable &strings,
    uint32_t max_lattice_size,
    GroupStringAnalysis &out_analysis,
    luisa::vector<TranslationError> &errors) noexcept;

// Pure string/format evaluation helpers (in string_table.cpp)
using HostFormatArg = luisa::variant<int32_t, float, luisa::float3, luisa::float4x4, luisa::string_view>;
[[nodiscard]] luisa::optional<luisa::string> eval_osl_format(
    luisa::string_view fmt,
    luisa::span<const HostFormatArg> args) noexcept;
[[nodiscard]] luisa::string eval_osl_substr(
    luisa::string_view s, int32_t start, int32_t len) noexcept;
[[nodiscard]] int32_t eval_osl_getchar(luisa::string_view s, int32_t index) noexcept;
[[nodiscard]] int32_t eval_osl_stoi(luisa::string_view s) noexcept;
[[nodiscard]] float eval_osl_stof(luisa::string_view s) noexcept;
[[nodiscard]] int32_t eval_osl_string_hash(luisa::string_view s) noexcept;
[[nodiscard]] luisa::vector<luisa::string> eval_osl_split(
    luisa::string_view s, luisa::string_view sep, int32_t maxsplit) noexcept;

// --- Shared opcode argument read/write classification ---

struct InstArgRW {
    uint64_t write_mask{0u};
    bool write_overflow{false};
    [[nodiscard]] constexpr bool writes_arg(size_t idx) const noexcept {
        return idx < 64u ? ((write_mask & (uint64_t{1u} << idx)) != 0u) : write_overflow;
    }
};
[[nodiscard]] InstArgRW classify_inst_rw(const Instruction &inst) noexcept;

// --- Derivative liveness analysis ---

struct GroupDerivAnalysis {
    // Per active layer: set of leaf symbols requiring Dual (val, dx, dy) storage
    luisa::vector<luisa::unordered_set<const Symbol *>> dual_symbols;
    size_t total_dual_symbols{0u};
};

void analyze_group_derivatives(
    const LinkedGroup &group,
    const GroupStringAnalysis &str_analysis,
    GroupDerivAnalysis &out_derivs) noexcept;

// --- Type & hint helpers ---

[[nodiscard]] bool is_simple_type(const Type *t, SimpleType::Primitive p) noexcept;
[[nodiscard]] bool is_triple_type(const Type *t) noexcept;
[[nodiscard]] bool is_differentiable_type(const Type *t) noexcept;
[[nodiscard]] size_t literal_stride(const Type *t) noexcept;
[[nodiscard]] const compute::Type *osl_type_to_xir_type(
    const Type *t, int array_len, const compute::Type *closure_type) noexcept;
[[nodiscard]] luisa::string_view unquote_hint_arg(luisa::string_view s) noexcept;
[[nodiscard]] luisa::optional<luisa::string_view> parse_argrw_hint(const Instruction &inst) noexcept;
[[nodiscard]] bool instruction_has_argderivs(const Instruction &inst, size_t arg_index_1based) noexcept;
[[nodiscard]] luisa::optional<GlobalSymbol> parse_global_symbol(luisa::string_view name) noexcept;
[[nodiscard]] bool is_global_writable(GlobalSymbol g) noexcept;
[[nodiscard]] bool is_global_differentiable(GlobalSymbol g) noexcept;
[[nodiscard]] luisa::float4x4 osl_row_major_to_xir_matrix(const float *vals) noexcept;

}// namespace luisa::compute::osl::detail

namespace luisa::compute::osl {

struct OSLTranslator::Impl {
    OSLTranslator *owner{nullptr};
    xir::Module *module{nullptr};
    xir::FunctionDefinition *function{nullptr};
    xir::XIRBuilder builder;
    TranslatorOptions options;
    StringTable string_table;
    luisa::vector<TranslationError> errors;

    detail::LinkedGroup linked_group;
    detail::GroupStringAnalysis string_analysis;
    detail::GroupDerivAnalysis deriv_analysis;

    // Current translation location for fail-closed error reporting
    luisa::string current_layer_name;
    uint32_t current_pc{~0u};
    luisa::string current_opcode;
    luisa::string current_filename;
    int current_line{-1};
    size_t string_switch_count{0u};

    // Per-active-layer symbol storage in XIR
    struct SymbolStorage {
        xir::Value *val_ptr{nullptr};// alloca for value (or uint32_t string token, or closure)
        xir::Value *dx_ptr{nullptr}; // nullptr if no derivs
        xir::Value *dy_ptr{nullptr};
        int array_length{0};         // effective length (>0 for arrays, 0 for scalar/empty)
        bool is_dual{false};
        bool is_string{false};
        bool is_closure{false};
        bool is_const{false};
    };

    struct LayerRuntime {
        luisa::unordered_map<const Symbol *, SymbolStorage> storage;
        xir::BasicBlock *layer_exit_block{nullptr};
    };
    luisa::vector<LayerRuntime> layer_runtimes;

    // Intra-group non-"trace" setmessage/getmessage slots keyed by message name
    struct MessageSlot {
        const Type *osl_type{nullptr};
        int array_length{0};
        luisa::vector<xir::Value *> val_ptrs;// leaf-flattened or single slot
        luisa::vector<xir::Value *> dx_ptrs;
        luisa::vector<xir::Value *> dy_ptrs;
        xir::Value *has_value_ptr{nullptr};  // bool alloca initialized false
    };
    luisa::unordered_map<luisa::string, MessageSlot> group_messages;

    // Constant cache helpers
    [[nodiscard]] xir::Value *const_int(int32_t v) noexcept;
    [[nodiscard]] xir::Value *const_uint(uint32_t v) noexcept;
    [[nodiscard]] xir::Value *const_float(float v) noexcept;
    [[nodiscard]] xir::Value *const_bool(bool v) noexcept;
    [[nodiscard]] xir::Value *const_float3(luisa::float3 v) noexcept;
    [[nodiscard]] xir::Value *const_float4(luisa::float4 v) noexcept;
    [[nodiscard]] xir::Value *const_float4x4(luisa::float4x4 v) noexcept;
    [[nodiscard]] xir::Value *zero_of(const compute::Type *t) noexcept;

    void set_location(const detail::LinkedLayer &layer, uint32_t pc, const Instruction &inst) noexcept;
    void clear_location() noexcept;
    void report_error(luisa::string msg) noexcept;

    // Value load/store/conversion helpers
    [[nodiscard]] const SymbolStorage *find_storage(uint32_t active_layer, const Symbol *sym) const noexcept;
    [[nodiscard]] int effective_array_length(uint32_t active_layer, const Symbol *sym) const noexcept;
    [[nodiscard]] DualValue load_symbol(uint32_t active_layer, const Symbol *sym, bool want_deriv) noexcept;
    void store_symbol(uint32_t active_layer, const Symbol *sym, const DualValue &val) noexcept;
    [[nodiscard]] DualValue load_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index, bool want_deriv) noexcept;
    void store_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index, const DualValue &val) noexcept;
    [[nodiscard]] ClosureValue load_closure(uint32_t active_layer, const Symbol *sym) noexcept;
    void store_closure(uint32_t active_layer, const Symbol *sym, const ClosureValue &c) noexcept;
    [[nodiscard]] ClosureValue load_closure_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index) noexcept;
    void store_closure_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index, const ClosureValue &c) noexcept;
    [[nodiscard]] xir::Value *load_string_token(uint32_t active_layer, const Symbol *sym) noexcept;
    void store_string_token(uint32_t active_layer, const Symbol *sym, xir::Value *token) noexcept;
    [[nodiscard]] xir::Value *load_string_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index) noexcept;
    void store_string_array_element(uint32_t active_layer, const Symbol *arr, xir::Value *index, xir::Value *token) noexcept;

    [[nodiscard]] xir::Value *to_bool(xir::Value *v) noexcept;
    [[nodiscard]] xir::Value *to_int(xir::Value *v) noexcept;
    [[nodiscard]] xir::Value *to_float(xir::Value *v) noexcept;
    [[nodiscard]] xir::Value *to_float3(xir::Value *v) noexcept;
    [[nodiscard]] xir::Value *to_float4x4(xir::Value *v) noexcept;
    [[nodiscard]] DualValue convert_dual(const DualValue &src, const Type *src_type, const Type *dst_type, bool want_deriv) noexcept;
    [[nodiscard]] xir::Value *extract_vec_comp(xir::Value *vec3, uint32_t idx) noexcept;
    [[nodiscard]] xir::Value *insert_vec_comp(xir::Value *vec3, xir::Value *elem, uint32_t idx) noexcept;
    [[nodiscard]] xir::Value *clamp_index(xir::Value *idx, int len) noexcept;
    [[nodiscard]] DualValue make_constant_dual(const Symbol *sym) noexcept;

    // Matrix helpers (XIR stores L = M^T)
    [[nodiscard]] xir::Value *transform_point_by_matrix(xir::Value *mat_L, xir::Value *pt) noexcept;
    [[nodiscard]] xir::Value *transform_vector_by_matrix(xir::Value *mat_L, xir::Value *vec) noexcept;
    [[nodiscard]] xir::Value *transform_normal_by_matrix(xir::Value *mat_L, xir::Value *nrm) noexcept;
    [[nodiscard]] DualValue transform_dual_by_matrix(
        xir::Value *mat_L, TransformKind kind, const DualValue &src, bool want_deriv) noexcept;

    // Dynamic string switch & named option dispatch
    luisa::unordered_map<uint32_t, luisa::string> external_string_reverse;
    [[nodiscard]] xir::Value *token_for_string(luisa::string_view sv) noexcept;
    [[nodiscard]] uint32_t external_id_for_string(luisa::string_view sv) noexcept;
    struct StringSlotSpec {
        detail::StringLattice lattice;
        xir::Value *runtime_token{nullptr};
    };
    void dispatch_dynamic_strings(
        luisa::span<const StringSlotSpec> slots,
        luisa::function<void(luisa::span<const luisa::string_view>)> emit_leaf) noexcept;

    // Translation phases
    void allocate_group_storage() noexcept;
    void emit_group_body(const ShaderGroup &group) noexcept;
    void emit_layer(uint32_t active_layer) noexcept;
    bool emit_code_range(uint32_t active_layer, const detail::OSOCodeRange &range,
                         xir::BasicBlock *next_after_range) noexcept;
    void emit_instruction(uint32_t active_layer, uint32_t pc, const Instruction &inst) noexcept;

    // Sub-translators (implemented in translator_ops.cpp, translator_color.cpp, translator_noise.cpp)
    [[nodiscard]] bool try_emit_instruction(uint32_t active_layer, uint32_t pc, const Instruction &inst) noexcept;
    [[nodiscard]] DualValue emit_default_color_transform(
        luisa::string_view from_space, luisa::string_view to_space,
        const DualValue &value, bool want_deriv) noexcept;
    [[nodiscard]] DualValue emit_default_noise(
        luisa::string_view noise_type, NoiseOp op,
        luisa::span<const DualValue> coords,
        luisa::span<const DualValue> periods,
        const Type *dst_type, bool want_deriv,
        luisa::span<const NamedOption> opts) noexcept;
    [[nodiscard]] DualValue emit_luminance(const DualValue &c, bool want_deriv) noexcept;
    [[nodiscard]] DualValue emit_blackbody(const DualValue &temp, bool want_deriv) noexcept;
    [[nodiscard]] DualValue emit_wavelength_color(const DualValue &lambda, bool want_deriv) noexcept;
    [[nodiscard]] xir::Value *emit_portable_erf(xir::Value *x) noexcept;
    [[nodiscard]] xir::Value *emit_portable_erfc(xir::Value *x) noexcept;
    [[nodiscard]] xir::Value *emit_portable_expm1(xir::Value *x) noexcept;

    // Intra-group message helpers
    void emit_intra_group_setmessage(
        luisa::string_view name, const Type *src_type, int src_array_len, const DualValue &value) noexcept;
    [[nodiscard]] xir::Value *emit_intra_group_getmessage(
        luisa::string_view name, const Type *dst_type, int dst_array_len, bool want_deriv, DualValue &out_value) noexcept;
};

}// namespace luisa::compute::osl
