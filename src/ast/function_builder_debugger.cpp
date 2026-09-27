// Debug function generator.
//
// This file implements an opt-in AST-level transform that rewrites a
// FunctionBuilder (kernel or callable) so that every statement/expression that
// can fail at runtime is wrapped with a guard: when the guard fires, a
// diagnostic is printed through the device printer
// (FunctionBuilder::print_, i.e. the DSL device_log path) and the thread
// stops early. Failures inside callables are reported to the caller through a
// generated trailing `reference<uint>` error out-parameter and propagate up
// to the root kernel, which stops the whole thread.
//
// ============================================================================
// Phase 1 — operator analysis (the definitive table of runtime-checkable
// issues that drives the generator; keep this in sync with include/luisa/ast/
// op.h and the per-check switches in DebugKernelOptions).
// ============================================================================
//
// UnaryOp (include/luisa/ast/op.h):
//   MINUS        int32/64 overflow when the operand is INT_MIN (edge case,
//                not checked in v1; float results are covered by NaN/Inf
//                checks on the *consumers* of the value).
//   BIT_NOT/NOT/logical not: no runtime error.
//
// BinaryOp (include/luisa/ast/op.h):
//   ADD/SUB/MUL  float operands/result: NaN/Inf detection on the result
//                (must-check). int operands: signed overflow (UB in C++,
//                wraps on GPUs; optional, off by default in v1).
//   DIV          int rhs == 0 -> division by zero (must-check). float rhs ==
//                0 -> Inf/NaN (must-check, reports the operands). float
//                NaN operand/result -> NaN/Inf result check (must-check).
//   MOD          int rhs == 0 -> division by zero (must-check); INT_MIN % -1
//                overflow (optional, not checked in v1). float MOD with a
//                zero divisor produces NaN -> covered by the NaN/Inf result
//                check.
//   SHL/SHR      shift amount >= bit-width of the shifted type or negative:
//                undefined value on GPUs (must-check for ints).
//   BIT_AND/BIT_OR/BIT_XOR/AND/OR: no runtime error.
//   Relational (LESS...NOT_EQUAL): no error; a NaN operand silently makes the
//                result false (no check in v1).
//
// CallOp groups with runtime issues:
//   * Math domain errors producing NaN/Inf (result check with ISNAN/ISINF;
//     must-check): SQRT/RSQRT (negative input), LOG/LOG2/LOG10 (input <= 0),
//     ACOS/ASIN (|x| > 1), ACOSH (x < 1), ATANH (|x| > 1), POW (negative base
//     with fractional exponent), ATAN2 (0, 0), EXP/EXP2/EXP10 (overflow ->
//     Inf), NORMALIZE (zero-length vector -> NaN), INVERSE (singular matrix).
//   * DETERMINANT, LENGTH*, DOT, CROSS, REFLECT, FACEFORWARD, FMA, COPYSIGN,
//     REDUCE_*, CLAMP, SATURATE, LERP, SMOOTHSTEP, STEP, ABS, MIN, MAX,
//     CLZ/CTZ/POPCOUNT/REVERSE, CEIL/FLOOR/FRACT/TRUNC/ROUND, ISINF/ISNAN,
//     SELECT, trigs, OUTER_PRODUCT, MATRIX_COMPONENT_WISE_MULTIPLICATION,
//     TRANSPOSE, RINT: no runtime error (trigs never trap on GPUs; RINT only
//     propagates an input NaN).
//   * Buffer ops (element-index bounds; must-check):
//       BUFFER_READ / BUFFER_VOLATILE_READ / BUFFER_WRITE /
//       BUFFER_VOLATILE_WRITE: index >= buffer size (CallOp::BUFFER_SIZE).
//       BYTE_BUFFER_READ/_VOLATILE_READ / BYTE_BUFFER_WRITE/
//       _VOLATILE_WRITE: byte_index + sizeof(T) > size
//       (CallOp::BYTE_BUFFER_SIZE).
//   * Texture ops: TEXTURE_READ/TEXTURE_WRITE coordinate bounds via
//     CallOp::TEXTURE_SIZE (opt-in: writes are clamped on some backends).
//   * Bindless (element range only; slot validity is a host-encoder contract
//     that lc::validation already enforces on the host):
//       BINDLESS/UNIFORM_/TYPED_/TYPED_UNIFORM_ BUFFER_READ/WRITE: elem_index
//         >= BINDLESS_BUFFER_SIZE(slot, sizeof(T)).
//       *_BYTE_BUFFER_READ: byte offset range with stride 1.
//       *TEXTURE2D/3D_SAMPLE*/READ*/SIZE*: slot check only (uv/coord are
//         hardware-clamped); no in-kernel check in v1.
//   * Atomics (ATOMIC_*): the address is an AtomicRef on an already-bound
//     buffer; no check.
//   * BUFFER_SIZE/BYTE_BUFFER_SIZE/BUFFER_ADDRESS/TEXTURE_SIZE/ACCEL_SIZE:
//     pure queries; no check.
//   * MAKE_*/ZERO/ONE/PACK/UNPACK/ASSERT/ASSUME/UNREACHABLE/FLATTEN/BRANCH/
//     FORCE_CASE/ADDRESS_OF/SYNCHRONIZE_BLOCK/CUSTOM/EXTERNAL/CLOCK/SHADER_
//     EXECUTION_REORDER/DDX/DDY/WARP_*/INDIRECT_SET_DISPATCH_*/TEXTURE2D_
//     SAMPLE (direct, hardware-clamped uv)/UNDEFINED: no check. ASSERT is its
//     own device-side guard; UNDEFINED is intentionally unspecified so
//     guarding it would change its semantics.
//   * Autodiff (REQUIRES_GRADIENT, GRADIENT, GRADIENT_MARKER,
//     ACCUMULATE_GRADIENT, BACKWARD, DETACH): transformation-time constructs.
//     The debug transform does not track gradients: functions whose CallOpSet
//     contains BACKWARD or GRADIENT are rejected with a clear error;
//     REQUIRES_GRADIENT/MARKER/DETACH pass through unguarded.
//   * Ray tracing instance access (must-check; requires CallOp::ACCEL_SIZE):
//       RAY_TRACING_INSTANCE_TRANSFORM / _USER_ID / _VISIBILITY_MASK,
//       RAY_TRACING_SET_INSTANCE_TRANSFORM / _VISIBILITY / _OPACITY /
//       _USER_ID, RAY_TRACING_INSTANCE_MOTION_MATRIX / _SRT,
//       RAY_TRACING_SET_INSTANCE_MOTION_MATRIX / _SRT: instance index >=
//       accel instance count reads/writes out of bounds on every backend
//       (CUDA accel.instances[index], HLSL instBuffer[index], DX fallback
//       inst[...]). The motion *key* is not part of ACCEL_SIZE; out-of-range
//       keys are not checked in v1 (host lc::validation bounds-checks
//       modification keys at build time).
//   * RAY_TRACING_TRACE_*/QUERY_* (+ motion blur variants): no instance index
//     argument. Ray validity (NaN/Inf origin/direction) is an opt-in check.
//   * RAY_QUERY_* object ops: operate on a RayQuery value produced by
//     RAY_TRACING_QUERY_*, well-formed by construction; no check.
//   * RASTER_DISCARD/RASTER_SET_Z_DEPTH: raster-stage functions are rejected
//     by the transform in v1.
//   * COOPERATIVE_*, ASYNC_COPY, PIPELINE_COMMIT/WAIT_PRIOR,
//     CLUSTER_LAUNCH_CONTROL_*, MBARRIER_*, FENCE_PROXY_ASYNC_*: expert
//     paths; functions using them are rejected with a clear error.
//
// ============================================================================
// Phase 2 — transform algorithm
// ============================================================================
// 1. Entry: the function is first canonicalized with FunctionBuilder::duplicate
//    (leaked cross-callable expressions are materialized there), then rebuilt
//    into a fresh builder with the same tag, block size, name and attributes.
//    Variables are remapped through a var_map, expressions through a
//    per-scope expr_map (a rebuilt node may be shared by several statements;
//    its guard/hoist statements must however be emitted in the scope that
//    first uses it, hence the per-scope cache).
// 2. Callable error-code ABI: every transformed callable gains a trailing
//    `reference<uint>` out-parameter (0 = ok, 1 = failed). Guards inside a
//    callable write 1 into it before returning; at transformed CUSTOM call
//    sites the caller creates a fresh local flag, passes it as the last
//    argument and checks it after the call: on failure it prints a diagnostic
//    and exits early (kernel) or propagates the error further (callable).
//    Non-void callables return an arbitrary value (CallOp::UNDEFINED, refined
//    to zero by backends) on the failure path; callers never use it because
//    they check the flag first. Kernels and void callables keep their ABI.
// 3. Statement lowering is a recursive scope walk that preserves order;
//    expressions are flattened post-order. A risky node is guarded *before*
//    it is evaluated (division/shift/bounds checks) or hoisted into a
//    temporary and checked *after* evaluation (NaN/Inf checks). Nodes without
//    a risky op and without hoisted children are rebuilt unchanged, so a
//    function without any checkable operation is rebuilt statement-for-
//    statement and expression-for-expression identical (zero overhead).
// 4. Expressions in loop for-init/cond/update positions and the provenance
//    while-condition of a `$while` loop are rebuilt WITHOUT guards: the AST
//    has no per-iteration statement position to host them (documented v1
//    limitation).

#include <luisa/ast/function_builder_debugger.h>

#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>

namespace luisa::compute::detail {

namespace {

// LUISA_AST_DEBUG_KERNEL / per-check overrides are read once per process.
[[nodiscard]] bool debug_env_flag(const char *name) noexcept {
    auto *value = std::getenv(name);
    if (value == nullptr) { return false; }
    auto flag = luisa::string_view{value};
    return flag == "1" || flag == "true" || flag == "TRUE" ||
           flag == "on" || flag == "ON";
}

[[nodiscard]] bool is_signed_int(const Type *t) noexcept {
    return t->is_int32() || t->is_int64() ||
           (t->is_vector() && t->element()->is_int32()) ||
           (t->is_vector() && t->element()->is_int64());
}

[[nodiscard]] bool is_integral(const Type *t) noexcept {
    if (t->is_vector()) { t = t->element(); }
    return t->is_int32() || t->is_uint32() || t->is_int64() || t->is_uint64() ||
           t->is_bool();
}

[[nodiscard]] bool is_floating_point(const Type *t) noexcept {
    if (t->is_vector()) { t = t->element(); }
    return t->is_float32() || t->is_float64();
}

// Element bit-width of a scalar/vector type.
[[nodiscard]] size_t element_bit_width(const Type *t) noexcept {
    auto elem = t->is_vector() ? t->element() : t;
    return elem->size() * 8u;
}

// Byte size of T rounded up to its alignment: the conservative footprint of
// one element in a byte-addressed buffer.
[[nodiscard]] size_t aligned_element_size(const Type *t) noexcept {
    auto size = t->size();
    auto alignment = t->alignment();
    return (size + alignment - 1u) / alignment * alignment;
}

}// namespace

class FunctionDebugger final {

public:
    struct Ctx {
        const FunctionBuilder &original;
        luisa::unordered_map<uint32_t /* original uid */,
                             const RefExpr * /* copy */>
            var_map{};
        // Rebuilt expressions, scoped to the scope that emitted their guard
        // statements: hoisted temporaries are only initialized along the
        // straight-line path of that scope.
        luisa::unordered_map<const Expression *, const Expression *> expr_map{};
        const RefExpr *err_ref{nullptr};
        const Type *ret_type{nullptr};
    };

    struct DebuggedFunction {
        luisa::shared_ptr<const FunctionBuilder> builder{};
        bool has_error_out{false};
    };

private:
    DebugKernelOptions _options;
    luisa::unordered_map<const FunctionBuilder *, DebuggedFunction> _debugged;
    luisa::vector<Ctx *> _contexts;

public:
    explicit FunctionDebugger(const DebugKernelOptions &options) noexcept
        : _options{options} {}

private:
    [[nodiscard]] Ctx &_ctx() noexcept { return *_contexts.back(); }
    [[nodiscard]] static FunctionBuilder *_fb() noexcept {
        return FunctionBuilder::current();
    }
    [[nodiscard]] luisa::string _func_name() noexcept {
        // The function under construction has no computed hash yet, so the
        // message names the (canonicalized) source function instead.
        return luisa::string{_ctx().original.debug_name()};
    }

    // ------------------------------------------------------------------
    // Literal/expression helpers. All of them build *new* expressions in
    // the builder under construction.
    // ------------------------------------------------------------------
    [[nodiscard]] const LiteralExpr *_uint_lit(uint32_t v) noexcept {
        return _fb()->literal(Type::of<uint32_t>(), LiteralExpr::Value{v});
    }
    [[nodiscard]] const LiteralExpr *_float_lit(float v) noexcept {
        return _fb()->literal(Type::of<float>(), LiteralExpr::Value{v});
    }
    [[nodiscard]] const Expression *
    _literal_of_type(const Type *t, uint64_t bits) noexcept {
        auto elem = t->is_vector() ? t->element() : t;
        LiteralExpr::Value value{};
        if (elem->is_int32()) {
            value = LiteralExpr::Value{static_cast<int32_t>(bits)};
        } else if (elem->is_uint32()) {
            value = LiteralExpr::Value{static_cast<uint32_t>(bits)};
        } else if (elem->is_int64()) {
            value = LiteralExpr::Value{static_cast<int64_t>(bits)};
        } else if (elem->is_uint64()) {
            value = LiteralExpr::Value{static_cast<uint64_t>(bits)};
        } else {
            LUISA_ASSERT(elem->is_float32() || elem->is_float64(),
                         "Unsupported literal type {}.", elem->description());
            value = LiteralExpr::Value{0.0f};
        }
        auto literal = _fb()->literal(elem, std::move(value));
        if (t->is_vector()) {
            luisa::vector<const Expression *> components(
                t->dimension(), literal);
            return _fb()->make_vector(t, components);
        }
        return literal;
    }
    [[nodiscard]] const Expression *_zero_value(const Type *t) noexcept {
        return _literal_of_type(t, 0u);
    }
    [[nodiscard]] const Expression *
    _uint_zero_value(const Type *t) noexcept {
        // zero of an integral (possibly vector) type
        auto elem = t->is_vector() ? t->element() : t;
        LiteralExpr::Value value{};
        if (elem->is_int64()) {
            value = LiteralExpr::Value{static_cast<int64_t>(0)};
        } else if (elem->is_uint64()) {
            value = LiteralExpr::Value{static_cast<uint64_t>(0)};
        } else if (elem->is_int32()) {
            value = LiteralExpr::Value{static_cast<int32_t>(0)};
        } else {
            value = LiteralExpr::Value{static_cast<uint32_t>(0)};
        }
        auto literal = _fb()->literal(elem, std::move(value));
        if (t->is_vector()) {
            luisa::vector<const Expression *> components(
                t->dimension(), literal);
            return _fb()->make_vector(t, components);
        }
        return literal;
    }
    [[nodiscard]] const Expression *
    _binary_bool(BinaryOp op, const Expression *lhs,
                 const Expression *rhs) noexcept {
        return _fb()->binary(Type::of<bool>(), op, lhs, rhs);
    }
    /// Reduce a (possibly vector) boolean expression to a scalar bool.
    [[nodiscard]] const Expression *
    _reduce_bool(const Expression *cond) noexcept {
        if (cond->type()->is_vector()) {
            return _fb()->call(Type::of<bool>(), CallOp::ANY, {cond});
        }
        return cond;
    }
    [[nodiscard]] const Expression *
    _cast_to_uint(const Expression *value) noexcept {
        auto t = value->type();
        if (t->is_vector()) {
            auto elem = t->element();
            if (elem->is_uint32()) { return value; }
            auto target = Type::vector(
                elem->is_int64() || elem->is_uint64() ?
                    Type::of<uint64_t>() :
                    Type::of<uint32_t>(),
                t->dimension());
            return _fb()->cast(target, CastOp::STATIC, value);
        }
        if (t->is_uint32()) { return value; }
        auto target = t->is_int64() || t->is_uint64() ?
                          Type::of<uint64_t>() :
                          Type::of<uint32_t>();
        return _fb()->cast(target, CastOp::STATIC, value);
    }

    // ------------------------------------------------------------------
    // Guard emission
    // ------------------------------------------------------------------
    /// Emit `$if (cond) { print(msg, args); <early exit> }`.
    void _emit_guard(const Expression *cond, luisa::string message,
                     luisa::vector<const Expression *> args) noexcept {
        auto fb = _fb();
        auto cond_reduced = _reduce_bool(cond);
        auto if_ = fb->if_(cond_reduced);
        fb->with(if_->true_branch(), [&] {
            fb->print_(std::move(message), args);
            _emit_early_exit();
        });
    }

    /// Stop the current thread: report the failure to the caller (callables)
    /// and return from the current function.
    void _emit_early_exit() noexcept {
        auto fb = _fb();
        auto &&ctx = _ctx();
        if (ctx.err_ref != nullptr) {
            fb->assign(ctx.err_ref, _uint_lit(1u));
        }
        if (ctx.ret_type != nullptr) {
            // The value is never consumed: the caller checks the error flag
            // first. UNDEFINED is refined to zero by the backends.
            luisa::vector<const Expression *> no_args;
            fb->return_(
                fb->call(ctx.ret_type, CallOp::UNDEFINED, no_args));
        } else {
            fb->return_();
        }
    }

    /// NaN/Inf condition of a float scalar/vector value.
    [[nodiscard]] const Expression *
    _nan_inf_cond(const Expression *value) noexcept {
        auto fb = _fb();
        auto t = value->type();
        auto bool_type = t->is_vector() ?
                             Type::vector(Type::of<bool>(), t->dimension()) :
                             Type::of<bool>();
        auto nan = fb->call(bool_type, CallOp::ISNAN, {value});
        auto inf = fb->call(bool_type, CallOp::ISINF, {value});
        return _binary_bool(BinaryOp::OR,
                            _reduce_bool(nan), _reduce_bool(inf));
    }

    /// Hoist `value` into a fresh local and return the local reference.
    [[nodiscard]] const RefExpr *_hoist(const Expression *value) noexcept {
        auto fb = _fb();
        auto local = fb->local(value->type());
        fb->assign(local, value);
        return local;
    }

    // ------------------------------------------------------------------
    // Check classification (Phase 1 table -> per-node checks)
    // ------------------------------------------------------------------
    static constexpr uint32_t check_none = 0u;
    static constexpr uint32_t check_int_div_mod_zero = 1u << 0u;
    static constexpr uint32_t check_float_div_zero = 1u << 1u;
    static constexpr uint32_t check_float_nan_inf = 1u << 2u;
    static constexpr uint32_t check_shift_amount = 1u << 3u;

    [[nodiscard]] uint32_t
    _binary_checks(BinaryOp op, const Type *lhs, const Type *rhs,
                   const Type *result) const noexcept {
        auto checks = check_none;
        switch (op) {
            case BinaryOp::DIV:
            case BinaryOp::MOD: {
                if (is_floating_point(result)) {
                    if (_options.check_float_div_zero) {
                        checks |= check_float_div_zero;
                    }
                    if (_options.check_float_nan_inf) {
                        checks |= check_float_nan_inf;
                    }
                } else if (is_integral(result) && _options.check_int_div_mod_zero) {
                    checks |= check_int_div_mod_zero;
                }
                break;
            }
            case BinaryOp::ADD:
            case BinaryOp::SUB:
            case BinaryOp::MUL: {
                if (is_floating_point(result) && _options.check_float_nan_inf) {
                    checks |= check_float_nan_inf;
                }
                break;
            }
            case BinaryOp::SHL:
            case BinaryOp::SHR: {
                if (is_integral(lhs) && _options.check_shift_amount) {
                    checks |= check_shift_amount;
                }
                break;
            }
            default: break;
        }
        return checks;
    }

    [[nodiscard]] bool _math_result_checked(CallOp op) const noexcept {
        if (!_options.check_float_nan_inf) { return false; }
        switch (op) {
            case CallOp::SQRT:
            case CallOp::RSQRT:
            case CallOp::LOG:
            case CallOp::LOG2:
            case CallOp::LOG10:
            case CallOp::ACOS:
            case CallOp::ASIN:
            case CallOp::ACOSH:
            case CallOp::ATANH:
            case CallOp::POW:
            case CallOp::ATAN2:
            case CallOp::EXP:
            case CallOp::EXP2:
            case CallOp::EXP10:
            case CallOp::NORMALIZE:
            case CallOp::INVERSE: return true;
            default: return false;
        }
    }

    // ------------------------------------------------------------------
    // Whole-function transform
    // ------------------------------------------------------------------
    [[nodiscard]] DebuggedFunction
    _debug(const FunctionBuilder &f) noexcept {
        if (auto iter = _debugged.find(&f); iter != _debugged.end()) {
            return iter->second;
        }
        _check_supported(f);
        auto has_error_out =
            _options.propagate_callable_errors &&
            f.tag() == Function::Tag::CALLABLE;
        DebuggedFunction result{nullptr, has_error_out};
        result.builder = FunctionBuilder::_define(f.tag(), [&] {
            Ctx ctx{
                .original = f,
                .var_map = {},
                .expr_map = {},
                .err_ref = nullptr,
                .ret_type = nullptr};
            _contexts.emplace_back(&ctx);
            _debug_function(f, has_error_out);
            LUISA_ASSERT(!_contexts.empty() && _contexts.back() == &ctx,
                         "Corrupted context stack.");
            _contexts.pop_back();
        });
        auto [iter, inserted] = _debugged.emplace(&f, result);
        LUISA_ASSERT(inserted, "FunctionDebugger::debug() called recursively.");
        return iter->second;
    }

    void _check_supported(const FunctionBuilder &f) const noexcept {
        switch (f.tag()) {
            case Function::Tag::KERNEL:
            case Function::Tag::CALLABLE: break;
            case Function::Tag::RASTER_STAGE:
                LUISA_ERROR(
                    "The debug function generator does not support "
                    "raster-stage functions (return-path rewrite is "
                    "ill-defined there).");
            case Function::Tag::COROUTINE:
                LUISA_ERROR(
                    "The debug function generator does not support "
                    "coroutines (return-path rewrite is ill-defined there).");
        }
        if (f.may_suspend()) {
            LUISA_ERROR(
                "The debug function generator does not support suspending "
                "functions (function '{}').",
                f.debug_name());
        }
        // Rejected expert paths. propagated_builtin_callables() already
        // contains everything reachable from this function.
        auto ops = f.direct_builtin_callables();
        ops.propagate(f.propagated_builtin_callables());
        auto reject = [&](CallOp op, luisa::string_view reason) {
            if (ops.test(op)) {
                LUISA_ERROR(
                    "The debug function generator does not support "
                    "function '{}' because it uses CallOp::{} ({}).",
                    f.debug_name(), magic_enum::enum_name(op), reason);
            }
        };
        reject(CallOp::BACKWARD, "autodiff backward pass");
        reject(CallOp::GRADIENT, "autodiff gradient");
        reject(CallOp::ASYNC_COPY, "async group copy");
        reject(CallOp::PIPELINE_COMMIT, "async copy pipeline control");
        reject(CallOp::PIPELINE_WAIT_PRIOR, "async copy pipeline control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL,
               "cluster launch control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL_MULTICAST,
               "cluster launch control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_IS_CANCELED,
               "cluster launch control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_X,
               "cluster launch control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Y,
               "cluster launch control");
        reject(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Z,
               "cluster launch control");
        reject(CallOp::MBARRIER_INIT, "mbarrier");
        reject(CallOp::MBARRIER_ARRIVE_EXPECT_TX, "mbarrier");
        reject(CallOp::MBARRIER_TRY_WAIT_PARITY, "mbarrier");
        reject(CallOp::FENCE_PROXY_ASYNC_ACQUIRE, "async proxy fence");
        reject(CallOp::FENCE_PROXY_ASYNC_RELEASE, "async proxy fence");
        if (f.use_cooperative_operations()) {
            LUISA_ERROR(
                "The debug function generator does not support function "
                "'{}' because it uses cooperative operations.",
                f.debug_name());
        }
        // Rejecting CallOps that to_string() does not cover would break the
        // message above; ACCEL_SIZE and the appended ops are all covered.
    }

    void _debug_function(const FunctionBuilder &f, bool has_error_out) noexcept {
        auto fb = _fb();
        auto &&ctx = _ctx();
        fb->mark_required_curve_basis_set(f.required_curve_bases());
        if (f.requires_noinline()) { fb->mark_noinline(); }
        fb->set_name(f.name());
        if (f.allowed_warp_size()) { fb->set_allowed_warp_size(*f.allowed_warp_size()); }
        if (f.tag() == Function::Tag::KERNEL) {
            fb->set_block_size(f.block_size());
        }
        ctx.ret_type = f.return_type();
        auto dup_arg = [&](Variable original) noexcept {
            auto dup = [&] {
                switch (original.tag()) {
                    case Variable::Tag::REFERENCE:
                        return fb->reference(original.type());
                    case Variable::Tag::BUFFER:
                        return fb->buffer(original.type());
                    case Variable::Tag::TEXTURE:
                        return fb->texture(original.type());
                    case Variable::Tag::BINDLESS_ARRAY:
                        return fb->bindless_array();
                    case Variable::Tag::ACCEL: return fb->accel();
                    default: return fb->argument(original.type());
                }
            }();
            ctx.var_map.emplace(original.uid(), dup);
        };
        if (f.tag() == Function::Tag::CALLABLE) {// captures are already lowered to arguments
            for (auto &arg : f.arguments()) { dup_arg(arg); }
        } else {
            for (auto i = 0u; i < f.bound_arguments().size(); i++) {
                auto &&a = f.arguments()[i];
                auto &&b = f.bound_arguments()[i];
                auto copy = luisa::visit(
                    [&]<typename B>(B &&bb) noexcept -> const RefExpr * {
                        using T = std::remove_cvref_t<B>;
                        if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                            return fb->buffer_binding(
                                a.type(), bb.handle, bb.offset, bb.size);
                        } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                            return fb->texture_binding(a.type(), bb.handle, bb.level);
                        } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                            return fb->bindless_array_binding(bb.handle);
                        } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                            return fb->accel_binding(bb.handle);
                        } else {
                            LUISA_ERROR_WITH_LOCATION("Unbound captured argument.");
                        }
                    },
                    b);
                ctx.var_map.emplace(a.uid(), copy);
            }
            for (auto &arg : f.unbound_arguments()) { dup_arg(arg); }
            for (auto &b : f.builtin_variables()) {
                auto copy = fb->_builtin(b.type(), b.tag());
                ctx.var_map.emplace(b.uid(), copy);
            }
        }
        for (auto &shared : f.shared_variables()) {
            auto s = fb->shared(shared.type());
            ctx.var_map.emplace(shared.uid(), s);
        }
        for (auto &local : f.local_variables()) {
            auto l = fb->local(local.type());
            ctx.var_map.emplace(local.uid(), l);
        }
        // Error-code ABI for callables: a trailing reference<uint>
        // out-parameter. It is created after every original argument so the
        // call sites (which append it last) match the callee's argument
        // order.
        if (has_error_out) {
            ctx.err_ref = fb->reference(Type::of<uint32_t>());
            fb->set_variable_name(
                ctx.err_ref->variable().uid(), "__dbg_err");
        }
        _debug_scope(f.body(), fb->body());
    }

    // ------------------------------------------------------------------
    // Statement lowering
    // ------------------------------------------------------------------
    void _debug_scope(const ScopeStmt *original, ScopeStmt *copy) noexcept {
        auto fb = _fb();
        auto &&ctx = _ctx();
        // Entering a scope invalidates hoisted temporaries emitted for the
        // enclosing scope's statements.
        luisa::unordered_map<const Expression *, const Expression *> saved;
        saved.swap(ctx.expr_map);
        fb->with(copy, [&] {
            for (auto s : original->statements()) {
                _debug_stmt(s);
                if (auto tag = s->tag();
                    tag == Statement::Tag::BREAK ||
                    tag == Statement::Tag::CONTINUE ||
                    tag == Statement::Tag::RETURN) {
                    break;
                }
            }
        });
        saved.swap(ctx.expr_map);
    }

    void _debug_stmt(const Statement *stmt) noexcept {
        auto fb = _fb();
        switch (stmt->tag()) {
            case Statement::Tag::BREAK: {
                fb->break_();
                break;
            }
            case Statement::Tag::CONTINUE: {
                fb->continue_();
                break;
            }
            case Statement::Tag::RETURN: {
                auto s = static_cast<const ReturnStmt *>(stmt);
                auto e = _flatten(s->expression());
                fb->return_(e);
                break;
            }
            case Statement::Tag::SCOPE: {
                LUISA_ERROR_WITH_LOCATION(
                    "ScopeStmt should have been handled in parent statements.");
            }
            case Statement::Tag::IF: {
                auto s = static_cast<const IfStmt *>(stmt);
                auto cond = _flatten(s->condition());
                auto if_ = fb->if_(cond);
                _debug_scope(s->true_branch(), if_->true_branch());
                _debug_scope(s->false_branch(), if_->false_branch());
                break;
            }
            case Statement::Tag::LOOP: {
                auto s = static_cast<const LoopStmt *>(stmt);
                auto loop = fb->loop_();
                auto condition = s->while_condition();
                auto rebuilt_condition =
                    condition == nullptr ?
                        nullptr :
                        _rebuild_unguarded(condition);
                _debug_scope(s->body(), loop->body());
                if (rebuilt_condition != nullptr) {
                    fb->mark_loop_as_while(
                        loop, rebuilt_condition,
                        s->while_condition_statement_count());
                }
                break;
            }
            case Statement::Tag::EXPR: {
                auto s = static_cast<const ExprStmt *>(stmt);
                auto e = _flatten(s->expression());
                // _flatten returns nullptr for void calls (they append their
                // own statement); _void_expr(nullptr) is a no-op.
                fb->_void_expr(e);
                break;
            }
            case Statement::Tag::SWITCH: {
                auto s = static_cast<const SwitchStmt *>(stmt);
                auto e = _flatten(s->expression());
                auto sw = fb->switch_(e);
                _debug_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_CASE:
            case Statement::Tag::SWITCH_CASE_GROUP: {
                auto s = static_cast<const SwitchCaseStmt *>(stmt);
                luisa::vector<const Expression *> labels;
                for (auto e : s->expressions()) {
                    labels.emplace_back(_flatten(e));
                }
                auto sw = fb->case_(luisa::span{labels});
                _debug_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_DEFAULT: {
                auto s = static_cast<const SwitchDefaultStmt *>(stmt);
                auto sw = fb->default_();
                _debug_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::ASSIGN: {
                auto s = static_cast<const AssignStmt *>(stmt);
                auto lhs = _flatten(s->lhs());
                auto rhs = _flatten(s->rhs());
                fb->assign(lhs, rhs);
                break;
            }
            case Statement::Tag::FOR: {
                auto s = static_cast<const ForStmt *>(stmt);
                // Guards cannot be expressed per-iteration at the AST level;
                // rebuild the for header without checks (documented v1
                // limitation).
                auto var = _rebuild_unguarded(s->variable());
                auto cond = _rebuild_unguarded(s->condition());
                auto step = _rebuild_unguarded(s->step());
                auto for_ = fb->for_(var, cond, step);
                _debug_scope(s->body(), for_->body());
                break;
            }
            case Statement::Tag::COMMENT: {
                auto s = static_cast<const CommentStmt *>(stmt);
                fb->comment_(luisa::string{s->comment()});
                break;
            }
            case Statement::Tag::SUSPEND: {
                LUISA_ERROR_WITH_LOCATION(
                    "The debug function generator does not support "
                    "suspending functions.");
            }
            case Statement::Tag::RAY_QUERY: {
                auto s = static_cast<const RayQueryStmt *>(stmt);
                auto q = _flatten(s->query());
                LUISA_ASSERT(q->tag() == Expression::Tag::REF,
                             "RayQueryExpr should be a reference.");
                auto rq = fb->ray_query_(static_cast<const RefExpr *>(q));
                _debug_scope(s->on_triangle_candidate(),
                             rq->on_triangle_candidate());
                _debug_scope(s->on_procedural_candidate(),
                             rq->on_procedural_candidate());
                break;
            }
            case Statement::Tag::AUTO_DIFF: {
                auto s = static_cast<const AutoDiffStmt *>(stmt);
                auto ad = fb->autodiff_();
                _debug_scope(s->body(), ad->body());
                break;
            }
            case Statement::Tag::PRINT: {
                auto s = static_cast<const PrintStmt *>(stmt);
                luisa::vector<const Expression *> args;
                args.reserve(s->arguments().size());
                for (auto arg : s->arguments()) {
                    args.emplace_back(_flatten(arg));
                }
                fb->print_(luisa::string{s->format()}, args);
                break;
            }
            case Statement::Tag::DEBUG_BREAK: {
                auto s = static_cast<const DebugBreakStmt *>(stmt);
                luisa::vector<const Expression *> watches;
                watches.reserve(s->watches().size());
                for (auto watch : s->watches()) {
                    watches.emplace_back(_flatten(watch));
                }
                fb->debug_break_(s->wrapper(), watches);
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Expression flattening + guard emission (the core)
    // ------------------------------------------------------------------
    const Expression *_flatten(const Expression *e) noexcept {
        if (e == nullptr) { return nullptr; }
        auto &&ctx = _ctx();
        if (auto iter = ctx.expr_map.find(e); iter != ctx.expr_map.end()) {
            return iter->second;
        }
        auto fb = _fb();
        const Expression *result = nullptr;
        switch (e->tag()) {
            case Expression::Tag::UNARY: {
                auto x = static_cast<const UnaryExpr *>(e);
                auto operand = _flatten(x->operand());
                result = fb->unary(x->type(), x->op(), operand);
                break;
            }
            case Expression::Tag::BINARY: {
                auto x = static_cast<const BinaryExpr *>(e);
                auto lhs = _flatten(x->lhs());
                auto rhs = _flatten(x->rhs());
                result = _flatten_binary(x, lhs, rhs);
                break;
            }
            case Expression::Tag::MEMBER: {
                auto x = static_cast<const MemberExpr *>(e);
                auto self = _flatten(x->self());
                result = x->is_swizzle() ?
                             static_cast<const Expression *>(
                                 fb->swizzle(x->type(), self,
                                             x->swizzle_size(),
                                             x->swizzle_code())) :
                             fb->member(x->type(), self, x->member_index());
                break;
            }
            case Expression::Tag::ACCESS: {
                auto x = static_cast<const AccessExpr *>(e);
                auto range = _flatten(x->range());
                auto index = _flatten(x->index());
                result = fb->access(x->type(), range, index);
                break;
            }
            case Expression::Tag::LITERAL: {
                auto x = static_cast<const LiteralExpr *>(e);
                result = fb->literal(x->type(), x->value());
                break;
            }
            case Expression::Tag::REF: {
                auto x = static_cast<const RefExpr *>(e);
                LUISA_ASSERT(x->builder() == &ctx.original,
                             "Leaked reference expression (from function "
                             "#{:016x}) inside function to be debugged; the "
                             "function should have been canonicalized with "
                             "FunctionBuilder::duplicate() first.",
                             x->builder()->hash());
                auto iter = ctx.var_map.find(x->variable().uid());
                LUISA_ASSERT(iter != ctx.var_map.end(),
                             "Variable not found in context.");
                result = iter->second;
                break;
            }
            case Expression::Tag::CONSTANT: {
                auto x = static_cast<const ConstantExpr *>(e);
                result = fb->constant(x->data());
                break;
            }
            case Expression::Tag::CALL: {
                auto x = static_cast<const CallExpr *>(e);
                result = _flatten_call(x);
                break;
            }
            case Expression::Tag::CAST: {
                auto x = static_cast<const CastExpr *>(e);
                auto src = _flatten(x->expression());
                result = fb->cast(x->type(), x->op(), src);
                break;
            }
            case Expression::Tag::TYPE_ID: {
                auto x = static_cast<const TypeIDExpr *>(e);
                result = fb->type_id(x->type());
                break;
            }
            case Expression::Tag::STRING_ID: {
                auto x = static_cast<const StringIDExpr *>(e);
                result = fb->string_id(luisa::string{x->data()});
                break;
            }
            case Expression::Tag::CPUCUSTOM: {
                auto x = static_cast<const CpuCustomOpExpr *>(e);
                result = fb->call(x->type(), x->func(), x->dtor(),
                                  x->user_data(), _flatten(x->arg()));
                break;
            }
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::FUNC_REF: {
                auto x = static_cast<const FuncRefExpr *>(e);
                auto f = _debug(*x->func());
                result = fb->func_ref(f.builder->function());
                break;
            }
        }
        ctx.expr_map.emplace(e, result);
        return result;
    }

    [[nodiscard]] const Expression *
    _flatten_binary(const BinaryExpr *x,
                    const Expression *lhs,
                    const Expression *rhs) noexcept {
        auto fb = _fb();
        auto checks = _binary_checks(
            x->op(), x->lhs()->type(), x->rhs()->type(), x->type());
        auto fname = _func_name();
        if (checks & check_int_div_mod_zero) {
            auto cond = _binary_bool(
                BinaryOp::EQUAL, rhs, _uint_zero_value(rhs->type()));
            _emit_guard(
                cond,
                luisa::format(
                    "lc-debug: integer division/modulo by zero in {} "
                    "(lhs={{}}, rhs={{}})",
                    fname),
                {lhs, rhs});
        }
        if (checks & check_float_div_zero) {
            auto cond = _binary_bool(
                BinaryOp::EQUAL, rhs, _zero_value(rhs->type()));
            _emit_guard(
                cond,
                luisa::format(
                    "lc-debug: float division by zero in {} "
                    "(lhs={{}}, rhs={{}})",
                    fname),
                {lhs, rhs});
        }
        if (checks & check_shift_amount) {
            auto rhs_type = rhs->type();
            auto bits = element_bit_width(x->lhs()->type());
            auto cond = _binary_bool(
                BinaryOp::GREATER_EQUAL, rhs,
                _literal_of_type(rhs_type, bits));
            if (is_signed_int(rhs_type)) {
                auto negative = _binary_bool(
                    BinaryOp::LESS, rhs, _zero_value(rhs_type));
                cond = _binary_bool(BinaryOp::OR, cond, negative);
            }
            _emit_guard(
                cond,
                luisa::format(
                    "lc-debug: shift amount out of range in {} (shift={{}})",
                    fname),
                {rhs});
        }
        auto node = fb->binary(x->type(), x->op(), lhs, rhs);
        if (checks & check_float_nan_inf) {
            auto local = _hoist(node);
            _emit_guard(
                _nan_inf_cond(local),
                luisa::format(
                    "lc-debug: NaN/Inf result in {} (binary op {}: "
                    "value={{}})",
                    fname, magic_enum::enum_name(x->op())),
                {local});
            return local;
        }
        return node;
    }

    [[nodiscard]] const Expression *
    _flatten_call(const CallExpr *x) noexcept {
        auto fb = _fb();
        auto fname = _func_name();
        if (x->is_custom()) {
            return _flatten_custom_call(x);
        }
        if (x->is_external()) {
            luisa::vector<const Expression *> args;
            args.reserve(x->arguments().size());
            for (auto arg : x->arguments()) {
                args.emplace_back(_flatten(arg));
            }
            auto &&externals = _ctx().original.external_callables();
            auto iter = std::find_if(
                externals.begin(), externals.end(),
                [ext = x->external()](auto &&f) noexcept {
                    return f.get() == ext;
                });
            LUISA_ASSERT(iter != externals.end(),
                         "External function not found in context.");
            return fb->call(x->type(), *iter, args);
        }
        auto op = x->op();
        auto arguments = x->arguments();
        luisa::vector<const Expression *> args;
        args.reserve(arguments.size());
        for (auto arg : arguments) {
            args.emplace_back(_flatten(arg));
        }
        // ---- pre-checks (must fire before the operation is evaluated) ----
        switch (op) {
            case CallOp::BUFFER_READ:
            case CallOp::BUFFER_VOLATILE_READ:
            case CallOp::BUFFER_WRITE:
            case CallOp::BUFFER_VOLATILE_WRITE: {
                if (_options.check_buffer_bounds) {
                    auto size = fb->call(
                        Type::of<uint32_t>(), CallOp::BUFFER_SIZE, {args[0]});
                    auto cond = _binary_bool(
                        BinaryOp::GREATER_EQUAL, _cast_to_uint(args[1]), size);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: buffer index out of range in {} "
                            "(index={{}}, size={{}})",
                            fname),
                        {args[1], size});
                }
                break;
            }
            case CallOp::BYTE_BUFFER_READ:
            case CallOp::BYTE_BUFFER_VOLATILE_READ:
            case CallOp::BYTE_BUFFER_WRITE:
            case CallOp::BYTE_BUFFER_VOLATILE_WRITE: {
                if (_options.check_buffer_bounds) {
                    auto size = fb->call(
                        Type::of<uint32_t>(), CallOp::BYTE_BUFFER_SIZE,
                        {args[0]});
                    auto offset = _cast_to_uint(args[1]);
                    // void writes have no result type: the element footprint
                    // comes from the written value instead.
                    auto elem_type = x->type() != nullptr ?
                                         x->type() :
                                         args[2]->type();
                    auto elem_bytes = static_cast<uint32_t>(
                        aligned_element_size(elem_type));
                    auto out_of_range = _binary_bool(
                        BinaryOp::GREATER_EQUAL, offset, size);
                    auto tail_too_small = _binary_bool(
                        BinaryOp::LESS,
                        fb->binary(Type::of<uint32_t>(), BinaryOp::SUB,
                                   size, offset),
                        _uint_lit(elem_bytes));
                    auto cond = _binary_bool(
                        BinaryOp::OR, out_of_range, tail_too_small);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: byte-buffer range out of bounds in {} "
                            "(offset={{}}, size={{}})",
                            fname),
                        {args[1], size});
                }
                break;
            }
            case CallOp::BINDLESS_BUFFER_READ:
            case CallOp::BINDLESS_BUFFER_WRITE:
            case CallOp::UNIFORM_BINDLESS_BUFFER_READ:
            case CallOp::UNIFORM_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_BINDLESS_BUFFER_READ:
            case CallOp::TYPED_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ:
            case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE: {
                if (_options.check_bindless_bounds) {
                    // void writes have no result type: the element footprint
                    // comes from the written value instead.
                    auto elem_type = x->type() != nullptr ?
                                         x->type() :
                                         args[3]->type();
                    auto stride = static_cast<uint32_t>(
                        aligned_element_size(elem_type));
                    auto count = fb->call(
                        Type::of<uint32_t>(),
                        _bindless_buffer_size_op(op),
                        {args[0], args[1], _uint_lit(stride)});
                    auto cond = _binary_bool(
                        BinaryOp::GREATER_EQUAL, _cast_to_uint(args[2]),
                        count);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: bindless element index out of range "
                            "in {} (elem_index={{}}, count={{}})",
                            fname),
                        {args[2], count});
                }
                break;
            }
            case CallOp::BINDLESS_BYTE_BUFFER_READ:
            case CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ:
            case CallOp::TYPED_BINDLESS_BYTE_BUFFER_READ:
            case CallOp::TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ: {
                if (_options.check_bindless_bounds) {
                    auto count = fb->call(
                        Type::of<uint32_t>(),
                        _bindless_buffer_size_op(op),
                        {args[0], args[1], _uint_lit(1u)});
                    auto offset = _cast_to_uint(args[2]);
                    // void writes have no result type: the element footprint
                    // comes from the written value instead.
                    auto elem_type = x->type() != nullptr ?
                                         x->type() :
                                         args[3]->type();
                    auto elem_bytes = static_cast<uint32_t>(
                        aligned_element_size(elem_type));
                    auto out_of_range = _binary_bool(
                        BinaryOp::GREATER_EQUAL, offset, count);
                    auto tail_too_small = _binary_bool(
                        BinaryOp::LESS,
                        fb->binary(Type::of<uint32_t>(), BinaryOp::SUB,
                                   count, offset),
                        _uint_lit(elem_bytes));
                    auto cond = _binary_bool(
                        BinaryOp::OR, out_of_range, tail_too_small);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: bindless byte-buffer range out of "
                            "bounds in {} (offset={{}}, size={{}})",
                            fname),
                        {args[2], count});
                }
                break;
            }
            case CallOp::TEXTURE_READ:
            case CallOp::TEXTURE_WRITE: {
                if (_options.check_texture_bounds) {
                    auto coord = args[1];
                    auto size = fb->call(coord->type(), CallOp::TEXTURE_SIZE,
                                         {args[0]});
                    auto coord_u = coord->type()->is_int32() ?
                                       static_cast<const Expression *>(fb->cast(
                                           Type::vector(Type::of<uint32_t>(),
                                                        coord->type()->dimension()),
                                           CastOp::STATIC, coord)) :
                                       coord;
                    auto cond = _binary_bool(
                        BinaryOp::GREATER_EQUAL, coord_u, size);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: texture coordinate out of range in {} "
                            "(coord={{}}, size={{}})",
                            fname),
                        {args[1], size});
                }
                break;
            }
            case CallOp::RAY_TRACING_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: {
                if (_options.check_accel_instance_index) {
                    auto count = fb->call(
                        Type::of<uint32_t>(), CallOp::ACCEL_SIZE, {args[0]});
                    auto cond = _binary_bool(
                        BinaryOp::GREATER_EQUAL, _cast_to_uint(args[1]),
                        count);
                    _emit_guard(
                        cond,
                        luisa::format(
                            "lc-debug: accel instance index out of range in "
                            "{} (index={{}}, count={{}})",
                            fname),
                        {args[1], count});
                }
                break;
            }
            case CallOp::RAY_TRACING_TRACE_CLOSEST:
            case CallOp::RAY_TRACING_TRACE_ANY:
            case CallOp::RAY_TRACING_QUERY_ALL:
            case CallOp::RAY_TRACING_QUERY_ANY:
            case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
            case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
            case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
            case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                if (_options.check_ray_validity) {
                    auto cond = _ray_validity_cond(args[1]);
                    if (cond != nullptr) {
                        _emit_guard(
                            cond,
                            luisa::format(
                                "lc-debug: non-finite ray in {}",
                                fname),
                            {});
                    }
                }
                break;
            }
            default: break;
        }
        // ---- the operation itself ----
        auto node = fb->call(
            x->type(), op, args, x->curve_basis_set());
        // ---- post-checks ----
        if (node != nullptr && is_floating_point(node->type()) &&
            _math_result_checked(op)) {
            auto local = _hoist(node);
            _emit_guard(
                _nan_inf_cond(local),
                luisa::format(
                    "lc-debug: NaN/Inf result in {} (call op {}: value={{}})",
                    fname, to_string(op)),
                {local});
            return local;
        }
        return node;
    }

    [[nodiscard]] static CallOp
    _bindless_buffer_size_op(CallOp op) noexcept {
        switch (op) {
            case CallOp::UNIFORM_BINDLESS_BUFFER_READ:
            case CallOp::UNIFORM_BINDLESS_BUFFER_WRITE:
            case CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ:
                return CallOp::UNIFORM_BINDLESS_BUFFER_SIZE;
            case CallOp::TYPED_BINDLESS_BUFFER_READ:
            case CallOp::TYPED_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_BINDLESS_BYTE_BUFFER_READ:
                return CallOp::TYPED_BINDLESS_BUFFER_SIZE;
            case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ:
            case CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE:
            case CallOp::TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ:
                return CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_SIZE;
            default: return CallOp::BINDLESS_BUFFER_SIZE;
        }
    }

    /// NaN/Inf condition over the origin/direction members of a ray.
    [[nodiscard]] const Expression *
    _ray_validity_cond(const Expression *ray) noexcept {
        auto t = ray->type();
        if (!t->is_structure() || t->members().size() < 3u) { return nullptr; }
        auto fb = _fb();
        const Expression *cond = nullptr;
        for (auto index : {0u, 2u}) {// origin, direction
            auto member = t->members()[index];
            if (!is_floating_point(member)) { return nullptr; }
            auto value = fb->member(member, ray, index);
            auto bad = _nan_inf_cond(value);
            cond = cond == nullptr ? bad :
                                     _binary_bool(BinaryOp::OR, cond, bad);
        }
        return cond;
    }

    [[nodiscard]] const Expression *
    _flatten_custom_call(const CallExpr *x) noexcept {
        auto fb = _fb();
        auto callee = _debug(*x->custom().builder());
        auto &&callee_builder = *callee.builder;
        auto fname = _func_name();
        luisa::vector<const Expression *> args;
        args.reserve(x->arguments().size() + 1u);
        for (auto arg : x->arguments()) {
            args.emplace_back(_flatten(arg));
        }
        const Expression *result = nullptr;
        if (callee.has_error_out) {
            auto err_flag = fb->local(Type::of<uint32_t>());
            fb->assign(err_flag, _uint_lit(0u));
            args.emplace_back(err_flag);
            result = fb->call(x->type(), callee_builder.function(), args);
            // A non-void call expression is materialized by its enclosing
            // statement, which would run *after* the check below; hoist it
            // into a temporary so the call happens before the check.
            if (result != nullptr) {
                result = _hoist(result);
            }
            auto failed = _binary_bool(
                BinaryOp::NOT_EQUAL, err_flag, _uint_lit(0u));
            _emit_guard(
                failed,
                luisa::format(
                    "lc-debug: callee {} failed in {}",
                    callee_builder.debug_name(), fname),
                {});
        } else if (x->type() != nullptr) {
            result = fb->call(x->type(), callee_builder.function(), args);
        } else {
            fb->call(callee_builder.function(), args);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Unguarded rebuild (for-loop headers, $while provenance conditions)
    // ------------------------------------------------------------------
    /// Rebuild an expression subtree without emitting any guard or hoisting
    /// any temporary and without touching the per-scope cache.
    const Expression *_rebuild_unguarded(const Expression *e) noexcept {
        if (e == nullptr) { return nullptr; }
        auto fb = _fb();
        auto &&ctx = _ctx();
        switch (e->tag()) {
            case Expression::Tag::UNARY: {
                auto x = static_cast<const UnaryExpr *>(e);
                return fb->unary(x->type(), x->op(),
                                 _rebuild_unguarded(x->operand()));
            }
            case Expression::Tag::BINARY: {
                auto x = static_cast<const BinaryExpr *>(e);
                return fb->binary(x->type(), x->op(),
                                  _rebuild_unguarded(x->lhs()),
                                  _rebuild_unguarded(x->rhs()));
            }
            case Expression::Tag::MEMBER: {
                auto x = static_cast<const MemberExpr *>(e);
                auto self = _rebuild_unguarded(x->self());
                return x->is_swizzle() ?
                           static_cast<const Expression *>(
                               fb->swizzle(x->type(), self,
                                           x->swizzle_size(),
                                           x->swizzle_code())) :
                           fb->member(x->type(), self, x->member_index());
            }
            case Expression::Tag::ACCESS: {
                auto x = static_cast<const AccessExpr *>(e);
                return fb->access(x->type(),
                                  _rebuild_unguarded(x->range()),
                                  _rebuild_unguarded(x->index()));
            }
            case Expression::Tag::LITERAL: {
                auto x = static_cast<const LiteralExpr *>(e);
                return fb->literal(x->type(), x->value());
            }
            case Expression::Tag::REF: {
                auto x = static_cast<const RefExpr *>(e);
                auto iter = ctx.var_map.find(x->variable().uid());
                LUISA_ASSERT(iter != ctx.var_map.end(),
                             "Variable not found in context.");
                return iter->second;
            }
            case Expression::Tag::CONSTANT: {
                auto x = static_cast<const ConstantExpr *>(e);
                return fb->constant(x->data());
            }
            case Expression::Tag::CAST: {
                auto x = static_cast<const CastExpr *>(e);
                return fb->cast(x->type(), x->op(),
                                _rebuild_unguarded(x->expression()));
            }
            case Expression::Tag::CALL: {
                auto x = static_cast<const CallExpr *>(e);
                luisa::vector<const Expression *> args;
                args.reserve(x->arguments().size());
                for (auto arg : x->arguments()) {
                    args.emplace_back(_rebuild_unguarded(arg));
                }
                if (x->is_builtin()) {
                    return fb->call(x->type(), x->op(), args,
                                    x->curve_basis_set());
                }
                if (x->is_custom()) {
                    auto callee = _debug(*x->custom().builder());
                    if (callee.has_error_out) {
                        LUISA_ERROR(
                            "The debug function generator requires callable "
                            "'{}' to be checked, but its result is used in "
                            "the unguarded header of a for loop.",
                            callee.builder->debug_name());
                    }
                    return fb->call(x->type(), callee.builder->function(),
                                    args);
                }
                LUISA_ASSERT(x->is_external(), "Unknown call type.");
                auto &&externals = _ctx().original.external_callables();
                auto iter = std::find_if(
                    externals.begin(), externals.end(),
                    [ext = x->external()](auto &&f) noexcept {
                        return f.get() == ext;
                    });
                LUISA_ASSERT(iter != externals.end(),
                             "External function not found in context.");
                return fb->call(x->type(), *iter, args);
            }
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported expression in an unguarded rebuild "
                    "(for-loop header / while provenance).");
        }
    }

public:
    [[nodiscard]] static DebuggedFunction
    debug(const FunctionBuilder &f, const DebugKernelOptions &options) noexcept {
        FunctionDebugger d{options};
        // Canonicalize first: the duplicator materializes expressions leaked
        // across callable boundaries into explicit arguments, so the rebuild
        // below only ever sees expressions owned by the function being
        // transformed.
        auto canonical = f.duplicate();
        return d._debug(*canonical);
    }
};

luisa::shared_ptr<const FunctionBuilder>
debug_function(const FunctionBuilder &f,
               const DebugKernelOptions &options) noexcept {
    return FunctionDebugger::debug(f, options).builder;
}

luisa::shared_ptr<const FunctionBuilder>
debug_function_if_enabled(luisa::shared_ptr<const FunctionBuilder> f) noexcept {
    static const auto enabled = debug_env_flag("LUISA_AST_DEBUG_KERNEL");
    if (!enabled || f == nullptr) { return f; }
    if (f->tag() != Function::Tag::KERNEL) { return f; }
    return debug_function(*f);
}

}// namespace luisa::compute::detail
