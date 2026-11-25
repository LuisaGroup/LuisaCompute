//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <cstdlib>
#include <limits>
#include <utility>

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/schedulers/detail/token_index.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/radix_sort.h>
#include <luisa/coro/schedulers/wavefront_auxiliary.h>
#include <luisa/coro/schedulers/wavefront_extension.h>
#include <luisa/coro/schedulers/wavefront_extension_batch.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute::coro {

namespace detail {
LUISA_CORO_API void coro_scheduler_compile_shaders(
    luisa::span<luisa::move_only_function<void()>> jobs, uint max_threads);
}// namespace detail

struct WavefrontCoroContinuationBlockSize {
    luisa::string continuation;
    uint execution_block_size = 0u;
};

struct WavefrontCoroSchedulerConfig {
    uint thread_count = static_cast<uint>(2_M);
    bool global_memory_soa = true;
    bool gather_by_sorting = true;
    bool frame_buffer_compaction = true;
    // Exclusive upper bound for every exported hint value.
    uint hint_range = 0xffffffffu;
    luisa::vector<luisa::string> hint_fields;
    bool report_stats = false;
    ShaderOption shader_option{};
    // Workgroup size for the thread-local generate/resume kernels. This is a
    // shader-structure choice and intentionally independent of the runtime
    // frame-pool capacity. Queue-management kernels retain their own block
    // sizes because some of them use block-local collectives. Kept last to
    // preserve the meaning of existing positional aggregate initializers.
    uint execution_block_size = 256u;
    // Cycles-style greedy scheduling executes only the largest non-empty
    // continuation queue in each host iteration. The default drains every
    // queue, preserving the original scheduler policy. This option changes
    // scheduling order only: coroutine transition semantics and frame layout
    // are identical. Kept after the legacy fields so positional aggregate
    // initializers retain their meaning.
    bool largest_continuation_first = false;
    // If non-empty, new entry work may be admitted alongside live frames only
    // when the largest continuation queue has one of these suspend names.
    // This models schedulers such as Cycles, which align old and new paths at
    // INTERSECT_CLOSEST before filling idle state slots. An empty list retains
    // the legacy unrestricted refill policy.
    luisa::vector<luisa::string> refill_continuations;
    // Minimum live-frame population below which refill is considered. Zero
    // selects half the active frame capacity, matching the legacy scheduler
    // and Cycles' upper bound for its device-derived busy-state threshold.
    uint refill_threshold = 0u;
    // Maintain the number of frames in each nonterminal continuation
    // incrementally at coroutine transitions, then scan only for the
    // continuation selected by the host. This replaces a complete token
    // multi-split per scheduling iteration with one selected-token gather.
    // LUISA_CORO_WAVEFRONT_VERIFY_QUEUES=1 independently materializes and
    // checks the invariant at scheduler boundaries. Without frame-buffer
    // compaction, free slots are gathered as an index queue at refill and
    // frames remain at stable addresses. This strategy requires greedy
    // one-continuation-at-a-time scheduling. Kept last for positional source
    // compatibility.
    bool incremental_continuation_counts = false;
    // Optional stable-frame locality partition for hint sorting. A nonzero
    // value sorts first by frame_index / hint_partition_size and then by the
    // exported hint. The key operands remain runtime shader arguments; only
    // the maximum composite range affects sort scratch/pass structure.
    uint hint_partition_size = 0u;
    // Optional per-continuation launch shapes. The default block size remains
    // the exact structure of the entry kernel and every continuation not
    // listed here. Overrides are keyed by semantic CoroGraph suspend name so
    // they are independent of graph node numbering. This is a host/JIT
    // specialization only; no selection enters generated device control flow.
    luisa::vector<WavefrontCoroContinuationBlockSize>
        continuation_block_sizes;
    // Publish incremental queue-count transitions in the entry and resume
    // kernels which produce them. This preserves the same finite-state count
    // conservation law as the standalone publisher kernels while avoiding a
    // second index/token pass. It is meaningful only together with
    // incremental_continuation_counts and is opt-in because it intentionally
    // changes user-kernel structure and resource pressure. Kept last for
    // positional aggregate source compatibility.
    bool fused_continuation_counts = false;
    // Host compilation only; does not change shader structure or scheduling.
    // Keep serial by default for callers/backends without concurrent compile
    // support. A small explicit limit bounds compiler memory consumption.
    uint shader_compile_threads = 1u;
};

/// Host-observed work executed by one coroutine graph node during the most
/// recent scheduler dispatch. Node zero is the entry generator; for every
/// other node, `executed_count` is exactly the number of frames submitted to
/// that continuation and `peak_queued_count` is the maximum materialized
/// queue cardinality observed at a scheduler boundary.
struct WavefrontCoroContinuationStats {
    size_t index{0u};
    size_t token{0u};
    luisa::string name;
    uint64_t dispatch_count{0u};
    uint64_t executed_count{0u};
    uint peak_queued_count{0u};
};

struct WavefrontCoroAuxiliaryStats {
    luisa::string name;
    uint64_t dispatch_count{0u};
    uint64_t executed_count{0u};
    uint peak_queued_count{0u};
};

struct WavefrontCoroExtensionStats {
    size_t queue_index{0u};
    size_t boundary_index{0u};
    size_t extension_index{0u};
    luisa::string schema;
    luisa::string handler;
    uint64_t dispatch_count{0u};
    uint64_t executed_count{0u};
    uint peak_queued_count{0u};
};

/// Diagnostics for the most recent dispatch. Collection is enabled by
/// WavefrontCoroSchedulerConfig::report_stats or
/// LUISA_CORO_WAVEFRONT_STATS. It is a host-only observation: enabling it
/// does not change generated shaders, coroutine frames, or queue semantics.
struct WavefrontCoroDispatchStats {
    bool collected{false};
    uint64_t iteration_count{0u};
    uint64_t generated_count{0u};
    uint64_t resumed_count{0u};
    uint64_t extension_count{0u};
    uint64_t gather_scan_count{0u};
    uint64_t compact_scan_count{0u};
    uint max_scan_count{0u};
    uint max_active_count{0u};
    double elapsed_ms{0.0};
    luisa::vector<WavefrontCoroContinuationStats> continuations;
    luisa::vector<WavefrontCoroExtensionStats> extensions;
    luisa::vector<WavefrontCoroAuxiliaryStats> auxiliary_work;
};

/// Stable host-side attribution for one scheduler-owned shader. The stage
/// label is semantic (entry, CoroGraph continuation, or queue operation), and
/// `structural_hash` is the exact Function hash seen by Device::compile. This
/// diagnostic metadata never enters the shader AST or cache identity.
struct WavefrontCoroShaderInfo {
    luisa::string stage;
    uint64_t structural_hash{0u};
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = WavefrontCoroSchedulerConfig;
    using ExtensionPrepareContext = WavefrontCoroExtensionPrepareContext;
    using ExtensionStage = WavefrontCoroExtensionStage;
    using ExtensionHandler = WavefrontCoroSchedulerExtensionHandler;

private:
    using AuxiliaryWork = WavefrontCoroAuxiliaryWork<Args...>;
    struct RegisteredAuxiliaryWork {
        luisa::shared_ptr<AuxiliaryWork> work;
        luisa::vector<uint> max_emitted_per_continuation;
        uint stage_count{0u};
        size_t stats_offset{0u};
    };
    struct RegisteredExtensionStage {
        WavefrontCoroExtensionStage stage;
        luisa::unique_ptr<ExtensionHandler> handler;
        uint next_queue{0u};
        luisa::vector<size_t> before_resume_chain;
    };
    struct ExtensionRoute {
        uint token{0u};
        uint boundary_index{0u};
        luisa::vector<size_t> source_store_fields;
    };

    Config _config;
    ByteBuffer _frame_buffer;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint, uint3, Args...>
        _gen_kernel;
    luisa::vector<Shader1D<uint, Buffer<uint>, uint, uint, uint, Args...>>
        _resume_kernels;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint>
        _advance_extension_stage_shader;
    Shader1D<uint, uint, Buffer<uint>> _initialize_shader;
    Shader1D<Buffer<uint>, uint> _clear_count_shader;
    Shader1D<uint, Buffer<uint>, uint> _count_shader;
    Shader1D<uint, Buffer<uint>, uint, Buffer<uint>, uint, uint>
        _publish_generated_count_shader;
    Shader1D<uint, Buffer<uint>, uint, Buffer<uint>, uint, uint>
        _publish_resumed_count_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint> _gather_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint, uint, uint>
        _gather_selected_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint, uint, uint, uint> _compact_shader;
    Shader1D<uint> _init_identity_shader;
    Buffer<uint> _resume_index;
    Buffer<uint> _identity_index;
    Buffer<uint> _resume_count;
    Buffer<uint> _resume_offset;
    Buffer<uint> _global_buffer;
    Buffer<uint> _extension_route_buffer;
    Buffer<uint> _sort_key[2];
    Buffer<uint> _sort_index;
    radix_sort::temp_storage _sort_temp_storage;
    radix_sort::instance<ByteBuffer, uint> _sort_token;
    radix_sort::instance<Buffer<uint>, ByteBuffer, uint, uint, uint>
        _sort_hint;
    luisa::vector<uint> _host_count;
    luisa::vector<uint> _host_offset;
    luisa::vector<bool> _have_hint;
    luisa::vector<bool> _refill_at;
    luisa::vector<bool> _is_terminal_continuation;
    CoroFrameStorageLayout _frame_layout;
    CoroFrameIOPlan _frame_io_plan;
    // Handlers and delayed finalization outlive the source Coroutine. Own
    // their immutable descriptors, not a borrowed compiler input or another
    // frame allocation. Declare these before handlers so they die last.
    CoroFrameDesc _extension_frame_desc;
    luisa::vector<CoroGraph::Boundary> _extension_boundaries;
    luisa::vector<RegisteredExtensionStage> _extension_stages;
    luisa::vector<luisa::vector<ExtensionRoute>> _extension_routes;
    luisa::vector<uint> _extension_route_table;
    // A physical queue can execute an annotation suffix and then a logical
    // continuation. Ordinary independent Extension queues have no such owner.
    luisa::vector<uint> _queue_continuation;
    luisa::vector<uint8_t> _queue_deferred;
    luisa::vector<uint8_t> _queue_deferred_self_loop;
    luisa::vector<uint> _queue_deferred_target;
    luisa::vector<uint8_t> _queue_blocks_deferred;
    luisa::vector<uint64_t> _shader_structure_hashes;
    luisa::vector<WavefrontCoroShaderInfo> _shader_infos;
    luisa::vector<uint> _continuation_block_sizes;
    WavefrontCoroDispatchStats _last_dispatch_stats;
    luisa::vector<RegisteredAuxiliaryWork> _auxiliary_work;
    Device *_device{nullptr};
    size_t _hint_field_index{static_cast<size_t>(-1)};
    uint _hint_key_range{0u};
    uint _continuation_count{0u};
    uint _queue_count{0u};
    uint _used_frame_count{0u};
    uint _active_frame_capacity{0u};
    uint _identity_initialized_count{0u};
    uint _deterministic_entry_target{0u};
    bool _has_hint_sort{false};
    bool _gather_selected_4x{false};
    bool _has_dispatched{false};
    bool _extension_handlers_finalized{false};

private:
    [[nodiscard]] static auto _linear_dispatch_index() noexcept {
        auto id = dispatch_id();
        auto size = dispatch_size();
        return id.x + id.y * size.x + id.z * size.x * size.y;
    }

    [[nodiscard]] static auto _dispatch_id_from_linear_index(UInt global_index, UInt3 dispatch_shape) noexcept {
        auto index_z = global_index / (dispatch_shape.x * dispatch_shape.y);
        auto index_xy = global_index - index_z * dispatch_shape.x * dispatch_shape.y;
        auto index_y = index_xy / dispatch_shape.x;
        auto index_x = index_xy - index_y * dispatch_shape.x;
        return make_uint3(index_x, index_y, index_z);
    }

    [[nodiscard]] static auto _next_power_of_two(uint value) noexcept {
        if (value <= 1u) { return 1u; }
        value--;
        value |= value >> 1u;
        value |= value >> 2u;
        value |= value >> 4u;
        value |= value >> 8u;
        value |= value >> 16u;
        return value + 1u;
    }

    [[nodiscard]] static auto _find_frame_field_index(const CoroFrameDesc &desc, luisa::string_view name) noexcept {
        auto index = desc.field_index(name);
        return index == static_cast<size_t>(-1) ?
                   index :
                   CoroFrameDesc::reserved_field_count + index;
    }

    [[nodiscard]] auto _valid_hint_field_count() const noexcept {
        auto count = 0u;
        for (auto hint : _have_hint) {
            if (hint) { count++; }
        }
        return count;
    }

    [[nodiscard]] static bool _is_target_token_member(
        const Expression *expr, uint frame_var_uid) noexcept {
        if (expr == nullptr || expr->tag() != Expression::Tag::MEMBER) {
            return false;
        }
        auto *member = static_cast<const MemberExpr *>(expr);
        if (member->is_swizzle() || member->member_index() != 6u ||
            member->self() == nullptr ||
            member->self()->tag() != Expression::Tag::REF) {
            return false;
        }
        return static_cast<const RefExpr *>(member->self())->variable().uid() == frame_var_uid;
    }

    [[nodiscard]] static bool _literal_equals_uint(
        const Expression *expr, uint expected) noexcept {
        if (expr == nullptr || expr->tag() != Expression::Tag::LITERAL) {
            return false;
        }
        auto *lit = static_cast<const LiteralExpr *>(expr);
        return luisa::visit(
            [expected]<typename T>(const T &v) noexcept -> bool {
                if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                    return static_cast<uint>(v) == expected;
                }
                return false;
            },
            lit->value());
    }

    [[nodiscard]] static bool _expr_references_var(
        const Expression *expr, uint var_uid) noexcept {
        if (expr == nullptr) { return false; }
        switch (expr->tag()) {
            case Expression::Tag::REF:
                return static_cast<const RefExpr *>(expr)->variable().uid() == var_uid;
            case Expression::Tag::UNARY:
                return _expr_references_var(static_cast<const UnaryExpr *>(expr)->operand(), var_uid);
            case Expression::Tag::BINARY: {
                auto *bin = static_cast<const BinaryExpr *>(expr);
                return _expr_references_var(bin->lhs(), var_uid) ||
                       _expr_references_var(bin->rhs(), var_uid);
            }
            case Expression::Tag::MEMBER:
                return _expr_references_var(static_cast<const MemberExpr *>(expr)->self(), var_uid);
            case Expression::Tag::ACCESS: {
                auto *acc = static_cast<const AccessExpr *>(expr);
                return _expr_references_var(acc->range(), var_uid) ||
                       _expr_references_var(acc->index(), var_uid);
            }
            case Expression::Tag::CALL: {
                auto *call = static_cast<const CallExpr *>(expr);
                for (auto *arg : call->arguments()) {
                    if (_expr_references_var(arg, var_uid)) { return true; }
                }
                return false;
            }
            case Expression::Tag::CAST:
                return _expr_references_var(static_cast<const CastExpr *>(expr)->expression(), var_uid);
            case Expression::Tag::LITERAL:
            case Expression::Tag::CONSTANT:
            case Expression::Tag::TYPE_ID:
            case Expression::Tag::STRING_ID:
            case Expression::Tag::FUNC_REF:
                return false;
            default:
                return true;
        }
    }

    struct _ScopeTokenAnalysis {
        bool always_returns{false};
        bool normal_expected{false};
        bool all_returns_ok{true};
    };

    [[nodiscard]] static _ScopeTokenAnalysis _analyze_scope_target_token(
        const ScopeStmt *scope, uint frame_var_uid,
        uint expected_token, bool definitely_expected) noexcept {
        if (scope == nullptr) {
            return {false, definitely_expected, true};
        }
        for (auto *stmt : scope->statements()) {
            if (stmt == nullptr) { continue; }
            switch (stmt->tag()) {
                case Statement::Tag::COMMENT:
                case Statement::Tag::DEBUG_BREAK:
                    break;
                case Statement::Tag::ASSIGN: {
                    auto *assign = static_cast<const AssignStmt *>(stmt);
                    if (_is_target_token_member(assign->lhs(), frame_var_uid)) {
                        if (_expr_references_var(assign->rhs(), frame_var_uid)) {
                            definitely_expected = false;
                        } else {
                            definitely_expected = _literal_equals_uint(assign->rhs(), expected_token);
                        }
                    } else if (_expr_references_var(assign->lhs(), frame_var_uid)) {
                        // A non-target_token member write might still clobber if whole-frame or aliased;
                        // only direct non-token frame member writes (MemberExpr != 6) preserve target_token.
                        bool safe_other_member = false;
                        if (assign->lhs()->tag() == Expression::Tag::MEMBER) {
                            auto *m = static_cast<const MemberExpr *>(assign->lhs());
                            if (!m->is_swizzle() && m->member_index() != 6u &&
                                m->self() != nullptr && m->self()->tag() == Expression::Tag::REF &&
                                static_cast<const RefExpr *>(m->self())->variable().uid() == frame_var_uid) {
                                safe_other_member = true;
                            }
                        }
                        if (!safe_other_member || _expr_references_var(assign->rhs(), frame_var_uid)) {
                            definitely_expected = false;
                        }
                    }
                    break;
                }
                case Statement::Tag::EXPR: {
                    auto *expr = static_cast<const ExprStmt *>(stmt)->expression();
                    if (_expr_references_var(expr, frame_var_uid)) {
                        definitely_expected = false;
                    }
                    break;
                }
                case Statement::Tag::SCOPE: {
                    auto inner = _analyze_scope_target_token(
                        static_cast<const ScopeStmt *>(stmt),
                        frame_var_uid, expected_token, definitely_expected);
                    if (!inner.all_returns_ok) { return {false, false, false}; }
                    if (inner.always_returns) { return {true, false, true}; }
                    definitely_expected = inner.normal_expected;
                    break;
                }
                case Statement::Tag::IF: {
                    auto *ifs = static_cast<const IfStmt *>(stmt);
                    if (_expr_references_var(ifs->condition(), frame_var_uid)) {
                        definitely_expected = false;
                    }
                    auto t = _analyze_scope_target_token(
                        ifs->true_branch(), frame_var_uid, expected_token, definitely_expected);
                    auto f = _analyze_scope_target_token(
                        ifs->false_branch(), frame_var_uid, expected_token, definitely_expected);
                    if (!t.all_returns_ok || !f.all_returns_ok) {
                        return {false, false, false};
                    }
                    if (t.always_returns && f.always_returns) {
                        return {true, false, true};
                    }
                    if (!t.always_returns && !f.always_returns) {
                        definitely_expected = t.normal_expected && f.normal_expected;
                    } else if (!t.always_returns) {
                        definitely_expected = t.normal_expected;
                    } else {
                        definitely_expected = f.normal_expected;
                    }
                    break;
                }
                case Statement::Tag::RETURN:
                    return {true, false, definitely_expected};
                default:
                    // Conservative fail-closed on loops, switches, or extension statements.
                    return {false, false, false};
            }
        }
        return {false, definitely_expected, true};
    }

    [[nodiscard]] uint _detect_deterministic_entry_target(const Coro &coro) const noexcept {
        if (!_config.incremental_continuation_counts ||
            !_config.fused_continuation_counts ||
            _continuation_count <= 1u ||
            _extension_routes.empty() ||
            !_extension_routes[0u].empty()) {
            return 0u;
        }
        const auto &entry_node = coro.graph().node(0u);
        if (entry_node.targets.size() != 1u) {
            return 0u;
        }
        const auto target = entry_node.targets.front();
        if (target == 0u || target >= _continuation_count) {
            return 0u;
        }
        auto entry_sub = coro.entry();
        if (!entry_sub) { return 0u; }
        Function wrapper = entry_sub.function();
        if (!wrapper || wrapper.body() == nullptr ||
            wrapper.body()->statements().size() != 1u ||
            wrapper.arguments().empty()) {
            return 0u;
        }
        auto *stmt = wrapper.body()->statements().front();
        if (stmt->tag() != Statement::Tag::EXPR) { return 0u; }
        auto *expr = static_cast<const ExprStmt *>(stmt)->expression();
        if (expr == nullptr || expr->tag() != Expression::Tag::CALL) { return 0u; }
        auto *call = static_cast<const CallExpr *>(expr);
        if (!call->is_custom() || call->arguments().empty() ||
            call->arguments().front()->tag() != Expression::Tag::REF) {
            return 0u;
        }
        auto wrapper_frame_uid = wrapper.arguments().front().uid();
        if (static_cast<const RefExpr *>(call->arguments().front())->variable().uid() != wrapper_frame_uid) {
            return 0u;
        }
        Function cc = call->custom();
        if (!cc || cc.body() == nullptr || cc.arguments().empty()) {
            return 0u;
        }
        auto cc_frame = cc.arguments().front();
        const auto expected_token = static_cast<uint>(coro.graph().node(target).token);
        if (expected_token == 0u || expected_token == CoroFrame::TERMINAL_TOKEN) {
            return 0u;
        }
        auto res = _analyze_scope_target_token(
            cc.body(), cc_frame.uid(), expected_token, false);
        if (res.all_returns_ok && (res.always_returns || res.normal_expected)) {
            return static_cast<uint>(target);
        }
        return 0u;
    }

    template<typename Kernel>
    [[nodiscard]] auto _compile_shader(
        Device &device, const Kernel &kernel,
        ShaderOption option, luisa::string stage) noexcept {
        auto structural_hash = kernel.function()->function().hash();
        _shader_structure_hashes.emplace_back(structural_hash);
        _shader_infos.emplace_back(WavefrontCoroShaderInfo{
            .stage = std::move(stage),
            .structural_hash = structural_hash});
        return detail::coro_scheduler_label_shader(
            device.compile(kernel, std::move(option)),
            _shader_infos.back().stage);
    }

    void _create_shader(Device &device, const Coro &coro) {
        _shader_structure_hashes.clear();
        _shader_infos.clear();
        auto nc = coro.subroutine_count();
        _continuation_count = static_cast<uint>(nc);
        _continuation_block_sizes.assign(
            nc, _config.execution_block_size);
        luisa::vector<bool> overridden(nc, false);
        for (auto &&override : _config.continuation_block_sizes) {
            auto node = coro.graph().node_by_name(
                override.continuation);
            LUISA_ASSERT(
                node != nullptr && node->index != 0u &&
                    node->index < nc,
                "Wavefront block-size override '{}' does not name a valid "
                "non-entry coroutine suspension.",
                override.continuation);
            LUISA_ASSERT(
                override.execution_block_size >= 32u &&
                    override.execution_block_size <= 1024u &&
                    override.execution_block_size % 32u == 0u,
                "Wavefront continuation '{}' block size must be a multiple "
                "of 32 in [32, 1024], but got {}.",
                override.continuation,
                override.execution_block_size);
            LUISA_ASSERT(
                !overridden[node->index],
                "Wavefront continuation '{}' has more than one block-size "
                "override.",
                override.continuation);
            overridden[node->index] = true;
            _continuation_block_sizes[node->index] =
                override.execution_block_size;
        }
        // This scheduler gathers and validates continuations through the
        // target-token field in global frame storage, so the token belongs to
        // its physical transfer plan.
        _frame_io_plan = coro_frame_make_io_plan(
            coro.graph(), coro.frame().frame_field_count(),
            {.externalize_target_token = true});
        _extension_stages.clear();
        _extension_routes.clear();
        _extension_routes.resize(nc);
        _last_dispatch_stats.extensions.clear();
        if (std::any_of(coro.graph().boundaries().begin(), coro.graph().boundaries().end(),
                        [](auto &&boundary) noexcept { return !boundary.extensions.empty(); })) {
            _extension_frame_desc = coro.frame();
            _extension_boundaries.reserve(coro.graph().boundary_count());
            for (auto &&source : coro.graph().boundaries()) {
                auto &boundary = _extension_boundaries.emplace_back(CoroGraph::Boundary{
                    .index = source.index, .from_index = source.from_index,
                    .to_index = source.to_index, .token = source.token,
                    .bindings = source.bindings, .source_store = source.source_store,
                    .target_live = source.target_live, .stages = source.stages});
                for (auto &&extension : source.extensions) {
                    boundary.extensions.emplace_back(extension->clone());
                }
            }
        }
        for (auto &&boundary : _extension_boundaries) {
            LUISA_ASSERT(
                boundary.from_index < nc && boundary.to_index < nc &&
                    boundary.stages.size() == boundary.extensions.size(),
                "Coroutine Extension boundary {} is inconsistent with its "
                "continuation graph.",
                boundary.index);
            if (boundary.stages.empty()) { continue; }
            auto &routes = _extension_routes[boundary.from_index];
            LUISA_ASSERT(
                std::none_of(
                    routes.begin(), routes.end(),
                    [&](auto &&route) noexcept {
                        return route.token == boundary.token;
                    }),
                "Continuation {} has several static Extension boundaries for "
                "token {}; a source-side route discriminator is required.",
                boundary.from_index, boundary.token);
            auto source_store_fields = boundary.source_store.slots;
            // Invocation identity is immutable scheduler ABI state and is not
            // part of Extension logical dataflow. Preserve precisely the
            // identity fields required by the eventual continuation while the
            // frame is resident in Extension queues.
            // Invocation identity is initialized exactly once by the entry
            // kernel and remains resident thereafter. A later continuation
            // need not load dormant identity fields into its local frame, so
            // writing those local fields on an Extension route would replace
            // the resident identity with zero. Entry routes must establish
            // the target's identity payload; all later routes preserve it and
            // let queue relocation copy the resident fields.
            if (boundary.from_index == coro.graph().entry_index()) {
                for (auto field :
                     _frame_io_plan.relocation_fields[boundary.to_index]) {
                    if (field <
                        CoroFrameDesc::reserved_field_count - 1u) {
                        coro_frame_append_unique(
                            source_store_fields, field);
                    }
                }
            }
            coro_frame_append_unique(source_store_fields, 6u);
            luisa::sort(source_store_fields.begin(),
                        source_store_fields.end());
            routes.emplace_back(ExtensionRoute{
                .token = static_cast<uint>(boundary.token),
                .boundary_index = static_cast<uint>(boundary.index),
                .source_store_fields = std::move(source_store_fields)});
            for (auto &&stage : boundary.stages) {
                LUISA_ASSERT(
                    stage.extension_index < boundary.extensions.size() &&
                        boundary.extensions[stage.extension_index] != nullptr,
                    "Coroutine Extension stage on boundary {} has invalid "
                    "Extension index {}.",
                    boundary.index, stage.extension_index);
                auto queue_index = nc + _extension_stages.size();
                auto *extension =
                    boundary.extensions[stage.extension_index].get();
                _extension_stages.emplace_back(RegisteredExtensionStage{
                    .stage = WavefrontCoroExtensionStage{
                        .queue_index = queue_index,
                        .boundary = &boundary,
                        .extension = extension,
                        .dataflow = &stage}});
                _last_dispatch_stats.extensions.emplace_back(
                    WavefrontCoroExtensionStats{
                        .queue_index = queue_index,
                        .boundary_index = boundary.index,
                        .extension_index = stage.extension_index,
                        .schema = luisa::string{extension->schema()}});
            }
        }
        _queue_count = static_cast<uint>(nc + _extension_stages.size());
        _queue_continuation.assign(_queue_count, static_cast<uint>(nc));
        for (auto i = 0u; i < nc; ++i) { _queue_continuation[i] = i; }
        _last_dispatch_stats.continuations.clear();
        _last_dispatch_stats.continuations.reserve(nc);
        for (auto i = 0u; i < nc; i++) {
            auto &&node = coro.graph().node(i);
            _last_dispatch_stats.continuations.emplace_back(
                WavefrontCoroContinuationStats{
                    .index = node.index,
                    .token = node.token,
                    .name = node.index == 0u ?
                                luisa::string{"<entry>"} :
                                node.name});
        }
        _resume_kernels.resize(nc);
        _host_count.resize(_queue_count);
        _host_offset.resize(_queue_count);
        _have_hint.resize(_queue_count, false);
        _refill_at.resize(_queue_count, false);
        _is_terminal_continuation.assign(nc, false);
        for (size_t i = 1u; i < nc; ++i) {
            _is_terminal_continuation[i] = coro.graph().node(i).targets.empty();
        }

        for (auto &name : _config.refill_continuations) {
            auto node = coro.graph().node_by_name(name);
            LUISA_ASSERT(node != nullptr && node->index != 0u &&
                             node->index < nc,
                         "Wavefront refill continuation '{}' does not name a "
                         "valid non-entry coroutine suspension.",
                         name);
            _refill_at[node->index] = true;
        }

        luisa::vector<luisa::string> valid_hint_fields;
        valid_hint_fields.reserve(_config.hint_fields.size());
        for (auto &name : _config.hint_fields) {
            if (auto node = coro.graph().node_by_name(name)) {
                if (node->index == 0u || node->index >= nc) {
                    LUISA_WARNING("Coroutine hint field '{}' resolves to invalid node {}; hint disabled.", name, node->index);
                } else {
                    _have_hint[node->index] = true;
                    valid_hint_fields.emplace_back(name);
                }
            } else {
                LUISA_WARNING("Coroutine hint field '{}' does not match a suspend name; hint disabled.", name);
            }
        }
        _config.hint_fields = std::move(valid_hint_fields);
        if (auto hint_count = _valid_hint_field_count(); hint_count != 0u) {
            _hint_field_index = _find_frame_field_index(coro.frame(), "coro_hint");
            if (_hint_field_index == static_cast<size_t>(-1)) {
                LUISA_WARNING("WavefrontCoroSchedulerConfig::hint_fields requires a uint frame value explicitly exported as 'coro_hint'; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            } else if (coro.frame().frame_field_type(_hint_field_index) != Type::of<uint>()) {
                LUISA_WARNING("Coroutine frame export 'coro_hint' must be uint; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            }
        }
        // Small hint ranges use the subgroup-independent bucket path. Larger
        // ranges use one-sweep radix sorting, whose rank construction is
        // explicitly defined for 32-lane subgroups. Hint sorting is only a
        // scheduling optimization, so disable that hint when the device does
        // not satisfy the algorithmic precondition; coroutine semantics and
        // token gathering remain unchanged.
        if (_valid_hint_field_count() != 0u) {
            LUISA_ASSERT(
                _config.hint_range != 0u,
                "Wavefront hint range must be a positive exclusive bound.");
            auto partition_count = 1u;
            if (_config.hint_partition_size != 0u) {
                partition_count =
                    _config.thread_count /
                        _config.hint_partition_size +
                    static_cast<uint>(
                        _config.thread_count %
                            _config.hint_partition_size !=
                        0u);
            }
            auto key_range =
                static_cast<uint64_t>(std::max(_config.hint_range, 1u)) *
                partition_count;
            LUISA_ASSERT(
                key_range <= std::numeric_limits<uint>::max(),
                "Wavefront composite hint range {} x {} exceeds uint.",
                _config.hint_range, partition_count);
            _hint_key_range = static_cast<uint>(key_range);
        }
        if (_valid_hint_field_count() != 0u &&
            _hint_key_range > radix_sort::hist_block_size &&
            device.compute_warp_size() != radix_sort::warp_size) {
            LUISA_WARNING(
                "Wavefront coroutine hint sorting over range {} requires "
                "{}-lane subgroups, but the device reports {}; hint sorting "
                "is disabled.",
                _hint_key_range, radix_sort::warp_size,
                device.compute_warp_size());
            std::fill(_have_hint.begin(), _have_hint.end(), false);
            _config.hint_fields.clear();
        }
        _has_hint_sort = _valid_hint_field_count() != 0u;
        auto use_token_sort =
            _config.gather_by_sorting &&
            !_config.incremental_continuation_counts;
        auto use_sort = use_token_sort || _has_hint_sort;

        auto queue_relocation_fields = _frame_io_plan.relocation_fields;
        queue_relocation_fields.reserve(_queue_count);
        for (auto &&registered : _extension_stages) {
            auto fields = registered.stage.dataflow->live_in.slots;
            auto target = registered.stage.boundary->to_index;
            for (auto field : _frame_io_plan.relocation_fields[target]) {
                if (field < CoroFrameDesc::reserved_field_count - 1u) {
                    coro_frame_append_unique(fields, field);
                }
            }
            coro_frame_append_unique(fields, 6u);
            luisa::sort(fields.begin(), fields.end());
            queue_relocation_fields.emplace_back(std::move(fields));
        }
        LUISA_ASSERT(
            queue_relocation_fields.size() == _queue_count,
            "Wavefront Extension queue relocation plan size mismatch.");

        if (!_config.global_memory_soa) {
            _frame_layout = CoroFrameStorageLayout::make_aos(
                coro.frame(), _config.thread_count);
        } else {
            const auto *disable_packed_env =
                std::getenv("LUISA_CORO_DISABLE_PACKED_SOA");
            const auto disable_packed =
                disable_packed_env != nullptr &&
                luisa::string_view{disable_packed_env} == "1";
            if (disable_packed) {
                _frame_layout = CoroFrameStorageLayout::make_runtime_soa(
                    coro.frame(), _config.thread_count);
            } else {
                luisa::vector<size_t> standalone_fields{6u};
                if (_has_hint_sort &&
                    _hint_field_index != static_cast<size_t>(-1)) {
                    coro_frame_append_unique(
                        standalone_fields, _hint_field_index);
                }
                luisa::vector<luisa::vector<size_t>> active_field_sets;
                for (size_t i = 1u; i < nc; ++i) {
                    if (!_frame_io_plan.input_fields[i].empty()) {
                        active_field_sets.emplace_back(
                            _frame_io_plan.input_fields[i]);
                    }
                }
                for (size_t s = 0u; s < nc; ++s) {
                    for (size_t t = 0u;
                         t < _frame_io_plan.transition_output_fields[s].size();
                         ++t) {
                        const auto &out =
                            _frame_io_plan.transition_output_fields[s][t];
                        if (!out.empty()) {
                            active_field_sets.emplace_back(out);
                        }
                    }
                    for (const auto &route : _extension_routes[s]) {
                        if (!route.source_store_fields.empty()) {
                            active_field_sets.emplace_back(
                                route.source_store_fields);
                        }
                    }
                }
                if (_config.frame_buffer_compaction) {
                    for (size_t q = 1u; q < queue_relocation_fields.size(); ++q) {
                        if (!queue_relocation_fields[q].empty()) {
                            active_field_sets.emplace_back(
                                queue_relocation_fields[q]);
                        }
                    }
                }
                for (const auto &boundary : _extension_boundaries) {
                    if (boundary.stages.empty()) { continue; }
                    for (const auto &stage : boundary.stages) {
                        if (!stage.reconstruct_slots.empty()) {
                            active_field_sets.emplace_back(
                                stage.reconstruct_slots);
                        }
                        if (!stage.def.slots.empty()) {
                            active_field_sets.emplace_back(stage.def.slots);
                        }
                        if (!stage.required_def.slots.empty()) {
                            active_field_sets.emplace_back(
                                stage.required_def.slots);
                        }
                    }
                    for (const auto &binding : boundary.bindings) {
                        auto slots = binding.reconstruct_slots();
                        if (slots.size() == 1u) {
                            coro_frame_append_unique(
                                standalone_fields, slots.front());
                        } else if (!slots.empty()) {
                            active_field_sets.emplace_back(
                                slots.begin(), slots.end());
                        }
                    }
                }
                _frame_layout = CoroFrameStorageLayout::make_packed_runtime_soa(
                    coro.frame(), _config.thread_count,
                    luisa::span<const luisa::vector<size_t>>{active_field_sets},
                    luisa::span<const size_t>{standalone_fields});
                if (_config.report_stats ||
                    std::getenv("LUISA_CORO_WAVEFRONT_STATS") != nullptr ||
                    std::getenv("LUISA_CORO_SHADER_MAP") != nullptr) {
                    size_t chunks4 = 0u;
                    size_t chunks2 = 0u;
                    for (const auto &chunk : _frame_layout.packed_chunks) {
                        if (chunk.width == 4u) { ++chunks4; }
                        else if (chunk.width == 2u) { ++chunks2; }
                    }
                    LUISA_INFO(
                        "Wavefront packed SoA layout: fields={} uint4_chunks={} "
                        "uint2_chunks={} packed_fields={} frame_stride={}B.",
                        coro.frame().frame_field_count(), chunks4, chunks2,
                        chunks4 * 4u + chunks2 * 2u, _frame_layout.frame_stride);
                }
            }
        }

        _frame_buffer = device.create_byte_buffer(_frame_layout.size_bytes);
        // The scheduler owns this complete allocation for the lifetime of all
        // generated shaders. Capturing it makes the zero-offset binding part
        // of the shader ABI (but not its structural hash), allowing backends
        // to prove the frame base alignment without weakening arbitrary
        // ByteBufferView semantics for user arguments.
        auto *frame_buffer = &_frame_buffer;
        _resume_index = device.create_buffer<uint>(_config.thread_count);
        if (_config.incremental_continuation_counts) {
            _identity_index = device.create_buffer<uint>(_config.thread_count);
        }
        _resume_count = device.create_buffer<uint>(_queue_count);
        auto *resume_count_buffer = &_resume_count;
        auto fuse_continuation_counts =
            _config.incremental_continuation_counts &&
            _config.fused_continuation_counts;
        // Incremental queue counts are scheduler-owned state. For
        // every nonterminal continuation t, C[t] is the cardinality of the
        // frames whose target token is t. Queue zero is derived from the
        // relevant slot domain minus sum(C[1..]) and is deliberately not
        // maintained.
        // The default policy binds this buffer only in scheduler-owned
        // publication kernels. The explicitly fused policy instead binds it
        // in producers so their transition atomics need no second pass.
        _resume_offset = device.create_buffer<uint>(_queue_count);
        _global_buffer = device.create_buffer<uint>(2u);
        if (!_extension_stages.empty()) {
            _extension_route_buffer = device.create_buffer<uint>(
                coro.graph().boundary_count());
        }
        auto *extension_route_buffer = &_extension_route_buffer;
        if (use_sort) {
            _sort_index = device.create_buffer<uint>(_config.thread_count);
            _sort_key[0] = device.create_buffer<uint>(_config.thread_count);
            _sort_key[1] = device.create_buffer<uint>(_config.thread_count);
            auto max_digit = std::max<uint>(
                _queue_count,
                std::min<uint>(_hint_key_range,
                               radix_sort::hist_block_size));
            _sort_temp_storage = radix_sort::temp_storage{device, _config.thread_count, max_digit};
        }

        auto token_to_index = detail::make_coro_token_index_callable(coro);
        Callable<uint(uint, ByteBuffer, uint)> read_scheduler_token = [layout = _frame_layout, soa = _config.global_memory_soa](
                                                                          UInt index, ByteBufferVar frame_buf,
                                                                          UInt frame_capacity) noexcept {
            return coro_frame_read_field<uint>(
                frame_buf, index, frame_capacity, layout, soa, 6u);
        };

        Callable<uint(uint, ByteBuffer, uint)> get_scheduler_token = [read_scheduler_token](
                                                                         UInt index, ByteBufferVar frame_buf,
                                                                         UInt frame_capacity) noexcept {
            return read_scheduler_token(index, frame_buf, frame_capacity);
        };
        Callable<uint(uint, ByteBuffer, uint)> identity_index = [](
                                                                    UInt index, ByteBufferVar,
                                                                    UInt) noexcept {
            return index;
        };
        if (use_token_sort) {
            _sort_token = radix_sort::instance<ByteBuffer, uint>{
                device, _config.thread_count, _sort_temp_storage,
                &get_scheduler_token, &identity_index, &get_scheduler_token,
                1u, _queue_count, 0u, 31u,
                "wavefront_token_sort"};
        }
        if (_has_hint_sort) {
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint, uint, uint)>
                keep_index = [](
                                 UInt index, BufferUInt values,
                                 ByteBufferVar, UInt, UInt, UInt) noexcept {
                    return values.read(index);
                };
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint, uint, uint)>
                get_coro_hint =
                    [layout = _frame_layout,
                     soa = _config.global_memory_soa,
                     hint_field_index = static_cast<uint>(_hint_field_index)](
                        UInt index, BufferUInt values,
                        ByteBufferVar frame_buf, UInt frame_capacity,
                        UInt hint_range, UInt partition_size) noexcept {
                        auto frame_index = values.read(index);
                        auto hint = coro_frame_read_field<uint>(
                            frame_buf, frame_index, frame_capacity,
                            layout, soa, hint_field_index);
                        auto partition = def(0u);
                        $if (partition_size != 0u) {
                            partition = frame_index / partition_size;
                        };
                        return hint + partition * hint_range;
                    };
            if (_hint_key_range <= radix_sort::hist_block_size) {
                auto hint_digit = std::max<uint>(_hint_key_range, 1u);
                _sort_hint = radix_sort::instance<
                    Buffer<uint>, ByteBuffer, uint, uint, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    1u, hint_digit, 0u, 31u,
                    "wavefront_hint_sort"};
            } else {
                auto high_bit = 0u;
                while ((_hint_key_range >> high_bit) != 1u) { high_bit++; }
                _sort_hint = radix_sort::instance<
                    Buffer<uint>, ByteBuffer, uint, uint, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    0u, radix_sort::hist_block_size, 0u, high_bit,
                    "wavefront_hint_sort"};
            }
        }

        const auto warp_aggregate_counts = device.compute_warp_size() == 32u;
        auto make_candidate_queues = [&](size_t node_index) noexcept {
            luisa::vector<uint> candidates;
            for (auto target : coro.graph().node(node_index).targets) {
                if (target != 0u && target < _queue_count) {
                    auto t = static_cast<uint>(target);
                    if (std::find(candidates.begin(), candidates.end(), t) ==
                        candidates.end()) {
                        candidates.emplace_back(t);
                    }
                }
            }
            if (!_extension_routes[node_index].empty()) {
                for (uint q = static_cast<uint>(nc); q < _queue_count; ++q) {
                    candidates.emplace_back(q);
                }
            }
            return candidates;
        };
        auto emit_fused_resume_counts =
            [resume_count_buffer, queue_count = _queue_count,
             warp_aggregate_counts](UInt next,
                                    const luisa::vector<uint> &candidates) noexcept {
                if (candidates.empty()) { return; }
                auto resume_count = Expr<Buffer<uint>>{*resume_count_buffer};
                if (warp_aggregate_counts) {
                    $if (warp_active_all_equal(next)) {
                        auto total = warp_active_count_bits(true);
                        $if ((next != 0u) & (next < queue_count) &
                             (warp_prefix_count_bits(true) == 0u)) {
                            resume_count.atomic(next).fetch_add(total);
                        };
                    } $else {
                        $if ((next != 0u) & (next < queue_count)) {
                            resume_count.atomic(next).fetch_add(1u);
                        };
                    };
                } else {
                    $if ((next != 0u) & (next < queue_count)) {
                        resume_count.atomic(next).fetch_add(1u);
                    };
                }
            };

        if (auto entry_sub = coro[0u]) {
            auto entry_candidates = make_candidate_queues(0u);
            Kernel1D k_gen = [&coro, frame_buffer, extension_route_buffer,
                              routes = _extension_routes[0u],
                              layout = _frame_layout, output_fields = _frame_io_plan.transition_output_fields[0u],
                              soa = _config.global_memory_soa, compact = _config.frame_buffer_compaction,
                              execution_block_size = _continuation_block_sizes[0u],
                              token_to_index, fuse_continuation_counts,
                              entry_candidates = std::move(entry_candidates),
                              emit_fused_resume_counts](
                                 UInt frame_capacity,
                                 BufferUInt resume_index,
                                 UInt index_offset, UInt frame_offset, UInt global_start,
                                 UInt count, UInt3 dispatch_shape,
                                 Var<Args>... k_args) noexcept {
                set_block_size(execution_block_size);
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto frame_id = compact ? frame_offset + x : resume_index.read(index_offset + x);
                auto logical_id = _dispatch_id_from_linear_index(global_start + x, dispatch_shape);
                auto frame = coro.instantiate(logical_id, dispatch_shape);
                frame.target_token = 0u;
                coro.entry()(frame, k_args...);
                auto raw_next = def(frame.target_token);
                auto next = def(token_to_index(raw_next));
                auto routed = def(false);
                if (!routes.empty()) {
                    auto route_table =
                        Expr<Buffer<uint>>{*extension_route_buffer};
                    for (auto &&route : routes) {
                        $if (raw_next == route.token) {
                            auto stage_queue =
                                route_table.read(route.boundary_index);
                            $if (stage_queue != 0u) {
                                next = stage_queue;
                                frame.target_token = next;
                                coro_frame_store(
                                    frame_buf, frame_id, frame_capacity, frame,
                                    layout, soa,
                                    luisa::span{route.source_store_fields},
                                    false, false);
                                routed = true;
                            };
                        };
                    }
                }
                frame.target_token = next;
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    if (output_fields[target].empty()) { continue; }
                    $if ((!routed) &
                         (next == static_cast<uint>(target))) {
                        coro_frame_store(
                            frame_buf, frame_id, frame_capacity, frame,
                            layout, soa, luisa::span{output_fields[target]},
                            false, false);
                    };
                }
                if (fuse_continuation_counts) {
                    emit_fused_resume_counts(next, entry_candidates);
                }
            };
            _gen_kernel = _compile_shader(
                device,
                k_gen,
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_generate"),
                "wavefront_generate/<entry>");
        }

        luisa::vector<luisa::move_only_function<void()>> pending_compilations;
        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            auto cont_candidates = make_candidate_queues(i);
            Kernel1D k_cont = [&coro, frame_buffer, extension_route_buffer,
                               resume_count_buffer,
                               routes = _extension_routes[i],
                               layout = _frame_layout, input_fields = _frame_io_plan.input_fields[i], output_fields = _frame_io_plan.transition_output_fields[i],
                               soa = _config.global_memory_soa, i,
                               execution_block_size = _continuation_block_sizes[i],
                               read_scheduler_token, token_to_index,
                               fuse_continuation_counts,
                               has_extensions = !_extension_stages.empty(),
                               queue_count = _queue_count,
                               cont_candidates = std::move(cont_candidates),
                               emit_fused_resume_counts](
                                  UInt frame_capacity,
                                  BufferUInt resume_index,
                                  UInt resume_offset, UInt count, UInt source_queue,
                                  Var<Args>... k_args) noexcept {
                set_block_size(execution_block_size);
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto idx = resume_index.read(resume_offset + x);
                auto tok = read_scheduler_token(idx, frame_buf, frame_capacity);
                auto current_queue = has_extensions ? source_queue : def(static_cast<uint>(i));
                if (fuse_continuation_counts) {
                    auto resume_count_buffer_expr =
                        Expr<Buffer<uint>>{*resume_count_buffer};
                    $if (x == 0u) {
                        resume_count_buffer_expr.atomic(
                            current_queue).fetch_sub(count);
                    };
                    $if (tok != current_queue) {
                        $if ((tok != 0u) & (tok < queue_count)) {
                            resume_count_buffer_expr.atomic(tok).fetch_add(1u);
                        };
                        $return();
                    };
                } else {
                    $if (tok != current_queue) { $return(); };
                }
                auto frame = coro_frame_load(
                    &coro.frame(), frame_buf, idx, frame_capacity,
                    layout, soa, luisa::span{input_fields}, false, false);
                frame.target_token = CoroFrame::TERMINAL_TOKEN;
                coro[i](frame, k_args...);
                auto raw_next = def(frame.target_token);
                auto next = def(token_to_index(raw_next));
                auto routed = def(false);
                if (!routes.empty()) {
                    auto route_table =
                        Expr<Buffer<uint>>{*extension_route_buffer};
                    for (auto &&route : routes) {
                        $if (raw_next == route.token) {
                            auto stage_queue =
                                route_table.read(route.boundary_index);
                            $if (stage_queue != 0u) {
                                next = stage_queue;
                                frame.target_token = next;
                                coro_frame_store(
                                    frame_buf, idx, frame_capacity, frame,
                                    layout, soa,
                                    luisa::span{route.source_store_fields},
                                    false, false);
                                routed = true;
                            };
                        };
                    }
                }
                frame.target_token = next;
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    if (output_fields[target].empty()) { continue; }
                    $if ((!routed) &
                         (next == static_cast<uint>(target))) {
                        coro_frame_store(
                            frame_buf, idx, frame_capacity, frame,
                            layout, soa, luisa::span{output_fields[target]},
                            false, false);
                    };
                }
                if (fuse_continuation_counts) {
                    emit_fused_resume_counts(next, cont_candidates);
                }
            };
            auto option = detail::coro_scheduler_shader_option(
                _config.shader_option,
                luisa::format("wavefront_resume_{}", i));
            auto stage = luisa::format("wavefront_resume_{}/{}", i,
                                       coro.graph().node(i).name);
            if (_config.shader_compile_threads <= 1u) {
                _resume_kernels[i] = _compile_shader(
                    device, k_cont, std::move(option), std::move(stage));
            } else {
                // Register diagnostics in graph order on the constructing
                // thread. Workers own the AST and label and write only their
                // preallocated shader slot; no shared vectors are appended.
                auto structural_hash = k_cont.function()->function().hash();
                _shader_structure_hashes.emplace_back(structural_hash);
                _shader_infos.emplace_back(WavefrontCoroShaderInfo{
                    .stage = stage, .structural_hash = structural_hash});
                pending_compilations.emplace_back(
                    [this, &device, i, kernel = std::move(k_cont),
                     option = std::move(option), stage = std::move(stage)]() mutable {
                        _resume_kernels[i] = detail::coro_scheduler_label_shader(
                            device.compile(kernel, std::move(option)), stage);
                    });
            }
        }
        detail::coro_scheduler_compile_shaders(
            pending_compilations, _config.shader_compile_threads);

        if (!_extension_stages.empty()) {
            Kernel1D advance_extension_stage =
                [frame_buffer, layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 read_scheduler_token, resume_count_buffer,
                 fuse_continuation_counts,
                 queue_count = _queue_count](
                    UInt frame_capacity, BufferUInt indices,
                    UInt index_offset, UInt count,
                    UInt current_queue, UInt next_queue) noexcept {
                    auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                    auto x = dispatch_x();
                    $if (x >= count) { $return(); };
                    auto frame_index = indices.read(index_offset + x);
                    auto token = read_scheduler_token(
                        frame_index, frame_buf, frame_capacity);
                    if (fuse_continuation_counts) {
                        auto resume_count =
                            Expr<Buffer<uint>>{*resume_count_buffer};
                        $if (x == 0u) {
                            resume_count.atomic(current_queue)
                                .fetch_sub(count);
                        };
                    }
                    $if (token == current_queue) {
                        coro_frame_write_field(
                            frame_buf, frame_index, frame_capacity,
                            layout, soa, 6u, next_queue);
                    };
                    if (fuse_continuation_counts) {
                        auto resume_count =
                            Expr<Buffer<uint>>{*resume_count_buffer};
                        auto final_token = select(
                            token, next_queue, token == current_queue);
                        $if ((final_token != 0u) &
                             (final_token < queue_count)) {
                            resume_count.atomic(final_token).fetch_add(1u);
                        };
                    }
                };
            _advance_extension_stage_shader = _compile_shader(
                device, advance_extension_stage,
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    "wavefront_advance_extension_stage"),
                "wavefront_advance_extension_stage");
        }

        Kernel1D initialize_kernel =
            [frame_buffer, layout = _frame_layout,
             soa = _config.global_memory_soa,
             initialize_free_indices =
                 _config.incremental_continuation_counts &&
                 !_config.frame_buffer_compaction](
                UInt frame_capacity, UInt n, BufferUInt free_index) {
            auto buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x < n) {
                coro_frame_write_field(
                    buf, x, frame_capacity, layout, soa, 6u, 0u);
                // Before the first admission every active slot is free, so
                // its index queue is the identity permutation. Materialize it
                // alongside token initialization and avoid a contended atomic
                // gather on the initial refill.
                if (initialize_free_indices) {
                    free_index.write(x, x);
                }
            };
        };
        _initialize_shader = _compile_shader(
            device, initialize_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_initialize"),
            "wavefront_initialize");

        if (_config.incremental_continuation_counts) {
            Kernel1D init_identity_kernel = [identity_buf = &_identity_index](UInt n) {
                auto x = dispatch_x();
                $if (x < n) { Expr<Buffer<uint>>{*identity_buf}.write(x, x); };
            };
            _init_identity_shader = _compile_shader(
                device, init_identity_kernel,
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_init_identity"),
                "wavefront_init_identity");
        }
        _deterministic_entry_target = _detect_deterministic_entry_target(coro);

        Kernel1D clear_count_kernel = [](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buffer.write(x, 0u); };
        };
        _clear_count_shader = _compile_shader(
            device, clear_count_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_clear_count"),
            "wavefront_clear_count");

        Kernel1D count_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, queue_count = _queue_count](
                UInt frame_capacity,
                BufferUInt count, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < queue_count) {
                count.atomic(tok).fetch_add(1u);
            };
        };
        _count_shader = _compile_shader(
            device, count_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_count"),
            "wavefront_count");

        if (_config.incremental_continuation_counts &&
            !_config.fused_continuation_counts) {
            // Queue cardinality is a scheduler concern, not continuation
            // state. Keep its atomics out of user coroutines so adding an
            // incremental scheduler cannot change a trace/shade kernel's
            // argument ABI, register pressure, or private-memory demand.
            //
            // Generation publishes
            //
            //   C' = C + histogram(target(generated_slot([0, n)))).
            //
            // With compaction, generated_slot(x) = frame_offset + x. With
            // stable frames it is the x-th index from the gathered free-slot
            // queue. Generation and publication use the same mapping.
            //
            // Token zero denotes termination and is intentionally omitted.
            Kernel1D publish_generated_count_kernel =
                [frame_buffer, layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 compact = _config.frame_buffer_compaction,
                 read_scheduler_token,
                 queue_count = _queue_count](
                    UInt frame_capacity, BufferUInt index,
                    UInt index_offset, BufferUInt count,
                    UInt frame_offset, UInt n) noexcept {
                    auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                    auto x = dispatch_x();
                    $if (x >= n) { $return(); };
                    auto frame_index = compact ?
                                           frame_offset + x :
                                           index.read(index_offset + x);
                    auto tok = read_scheduler_token(
                        frame_index, frame_buf, frame_capacity);
                    $if ((tok != 0u) & (tok < queue_count)) {
                        count.atomic(tok).fetch_add(1u);
                    };
                };
            _publish_generated_count_shader = _compile_shader(
                device, publish_generated_count_kernel,
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    "wavefront_publish_generated_count"),
                "wavefront_publish_generated_count");

            // Resuming the complete selected queue i with index set S obeys
            // the exact finite-state conservation law
            //
            //   C' = C - |S| e_i + sum(f in S) e_target(f).
            //
            // The subtraction and histogram increments are commutative
            // atomics. This remains correct for self-edges (target == i),
            // sparse source tokens (the scheduler uses dense node indices),
            // and frames that fail the defensive token check (their old i is
            // read back and restored by the histogram).
            Kernel1D publish_resumed_count_kernel =
                [frame_buffer, layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 read_scheduler_token,
                 queue_count = _queue_count](
                    UInt frame_capacity, BufferUInt index,
                    UInt index_offset, BufferUInt count,
                    UInt source_token, UInt n) noexcept {
                    auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                    auto x = dispatch_x();
                    $if (x >= n) { $return(); };
                    $if (x == 0u) {
                        count.atomic(source_token).fetch_sub(n);
                    };
                    auto frame_index = index.read(index_offset + x);
                    auto tok = read_scheduler_token(
                        frame_index, frame_buf, frame_capacity);
                    $if ((tok != 0u) & (tok < queue_count)) {
                        count.atomic(tok).fetch_add(1u);
                    };
                };
            _publish_resumed_count_shader = _compile_shader(
                device, publish_resumed_count_kernel,
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    "wavefront_publish_resumed_count"),
                "wavefront_publish_resumed_count");
        }

        Kernel1D gather_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, queue_count = _queue_count](
                UInt frame_capacity,
                BufferUInt index, BufferUInt offset, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < queue_count) {
                auto slot = offset.atomic(tok).fetch_add(1u);
                index.write(slot, x);
            };
        };
        _gather_shader = _compile_shader(
            device, gather_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_gather"),
            "wavefront_gather");

        if (_config.incremental_continuation_counts) {
            if (device.compute_warp_size() == 32u) {
                _gather_selected_4x = true;
                Kernel1D gather_selected_kernel =
                    [frame_buffer, layout = _frame_layout,
                     soa = _config.global_memory_soa,
                     read_scheduler_token](
                        UInt frame_capacity, BufferUInt index,
                        BufferUInt write_count, UInt count_slot, UInt selected, UInt n) noexcept {
                        set_block_size(256u);
                        set_warp_size(32u);
                        Shared<uint> warp_totals{32u};
                        Shared<uint> block_base{1u};
                        auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                        $if(dispatch_x() == 0u) { write_count.write(count_slot ^ 1u, 0u); };
                        auto tid = thread_x();
                        auto warp_id = tid >> 5u;
                        auto lane_id = tid & 31u;
                        auto block_start = block_x() * 1024u + tid;
                        auto idx0 = block_start;
                        auto idx1 = block_start + 256u;
                        auto idx2 = block_start + 512u;
                        auto idx3 = block_start + 768u;
                        auto active0 = def(false);
                        auto active1 = def(false);
                        auto active2 = def(false);
                        auto active3 = def(false);
                        $if(idx0 < n) { active0 = read_scheduler_token(idx0, frame_buf, frame_capacity) == selected; };
                        $if(idx1 < n) { active1 = read_scheduler_token(idx1, frame_buf, frame_capacity) == selected; };
                        $if(idx2 < n) { active2 = read_scheduler_token(idx2, frame_buf, frame_capacity) == selected; };
                        $if(idx3 < n) { active3 = read_scheduler_token(idx3, frame_buf, frame_capacity) == selected; };
                        auto p0 = warp_prefix_count_bits(active0);
                        auto c0 = warp_active_count_bits(active0);
                        auto p1 = warp_prefix_count_bits(active1);
                        auto c1 = warp_active_count_bits(active1);
                        auto p2 = warp_prefix_count_bits(active2);
                        auto c2 = warp_active_count_bits(active2);
                        auto p3 = warp_prefix_count_bits(active3);
                        auto c3 = warp_active_count_bits(active3);
                        $if(lane_id == 0u) {
                            warp_totals[warp_id] = c0;
                            warp_totals[8u + warp_id] = c1;
                            warp_totals[16u + warp_id] = c2;
                            warp_totals[24u + warp_id] = c3;
                        };
                        sync_block();
                        $if(tid < 32u) {
                            auto wc = warp_totals[tid];
                            auto wp = warp_prefix_sum(wc);
                            auto total = warp_active_sum(wc);
                            warp_totals[tid] = wp;
                            $if((tid == 0u) & (total != 0u)) {
                                block_base[0u] = write_count.atomic(count_slot).fetch_add(total);
                            };
                        };
                        sync_block();
                        auto base_out = block_base[0u];
                        $if(active0) { index.write(base_out + warp_totals[warp_id] + p0, idx0); };
                        $if(active1) { index.write(base_out + warp_totals[8u + warp_id] + p1, idx1); };
                        $if(active2) { index.write(base_out + warp_totals[16u + warp_id] + p2, idx2); };
                        $if(active3) { index.write(base_out + warp_totals[24u + warp_id] + p3, idx3); };
                    };
                _gather_selected_shader = _compile_shader(
                    device, gather_selected_kernel,
                    detail::coro_scheduler_shader_option(
                        _config.shader_option, "wavefront_gather_selected"),
                    "wavefront_gather_selected");
            } else {
                Kernel1D gather_selected_kernel =
                    [frame_buffer, layout = _frame_layout,
                     soa = _config.global_memory_soa,
                     read_scheduler_token](
                        UInt frame_capacity, BufferUInt index,
                        BufferUInt write_count, UInt count_slot, UInt selected, UInt n) noexcept {
                        auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                        auto x = dispatch_x();
                        $if(x == 0u) { write_count.write(count_slot ^ 1u, 0u); };
                        $if(x >= n) { $return(); };
                        auto tok = read_scheduler_token(
                            x, frame_buf, frame_capacity);
                        $if(tok == selected) {
                            auto slot = write_count.atomic(count_slot).fetch_add(1u);
                            index.write(slot, x);
                        };
                    };
                _gather_selected_shader = _compile_shader(
                    device, gather_selected_kernel,
                    detail::coro_scheduler_shader_option(
                        _config.shader_option, "wavefront_gather_selected"),
                    "wavefront_gather_selected");
            }
        }

        auto relocation_partition =
            coro_frame_partition_relocation_fields(
                std::move(queue_relocation_fields),
                _config.global_memory_soa ? &_frame_layout : nullptr);
        Kernel1D compact_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, desc = &coro.frame(),
             queue_count = _queue_count,
             incremental = _config.incremental_continuation_counts,
             warp_aggregate_counts,
             common_relocation_fields =
                 std::move(relocation_partition.common_fields),
             residual_relocation_fields =
                 std::move(relocation_partition.residual_fields)](
                UInt frame_capacity,
                BufferUInt index, BufferUInt global, UInt count_slot,
                UInt active_count, UInt empty_count, UInt scan_count) noexcept {
                if (incremental && warp_aggregate_counts) {
                    set_block_size(256u);
                    set_warp_size(32u);
                }
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                $if (x == 0u) { global.write(count_slot ^ 1u, 0u); };
                auto src = active_count + x;
                auto relocate_frame = [&](UInt tok, UInt dst) noexcept {
                    // A queued frame is the tagged union Sigma_t R[t].
                    // Copying every physical slot would read outside the
                    // domain R[t], where values are intentionally
                    // undefined after interference coloring. The common
                    // intersection plus the token residual is exactly
                    // R[t], as certified by CoroGraph's fixed-point
                    // live-begin projection.
                    if (!common_relocation_fields.empty()) {
                        coro_frame_copy_fields(
                            frame_buf, src, dst, frame_capacity, desc,
                            layout, soa,
                            luisa::span<const size_t>{
                                common_relocation_fields});
                    }
                    auto copy_residual = switch_(tok);
                    for (size_t token = 1u;
                         token < residual_relocation_fields.size();
                         ++token) {
                        copy_residual = std::move(copy_residual).case_(
                            static_cast<uint>(token), [&, token] {
                                auto &fields =
                                    residual_relocation_fields[token];
                                if (!fields.empty()) {
                                    coro_frame_copy_fields(
                                        frame_buf, src, dst,
                                        frame_capacity, desc, layout,
                                        soa,
                                        luisa::span<const size_t>{
                                            fields});
                                }
                            });
                    }
                    std::move(copy_residual).default_([] {});
                    coro_frame_write_field(
                        frame_buf, src, frame_capacity,
                        layout, soa, 6u, 0u);
                };
                if (incremental) {
                    auto tok = def(0u);
                    $if (src < scan_count) {
                        tok = read_scheduler_token(src, frame_buf, frame_capacity);
                    };
                    auto is_active_src = (tok != 0u) & (tok < queue_count);
                    if (warp_aggregate_counts) {
                        auto lane_prefix = warp_prefix_count_bits(is_active_src);
                        auto warp_total = warp_active_count_bits(is_active_src);
                        auto warp_base = def(0u);
                        $if (((thread_x() & 31u) == 0u) & (warp_total != 0u)) {
                            warp_base = global.atomic(count_slot).fetch_add(warp_total);
                        };
                        warp_base = warp_read_lane(warp_base, 0u);
                        $if (is_active_src) {
                            auto res = warp_base + lane_prefix;
                            $if (res < empty_count) {
                                auto dst = index.read(res);
                                $if (dst < active_count) {
                                    relocate_frame(tok, dst);
                                };
                            };
                        };
                    } else {
                        $if (is_active_src) {
                            auto res = global.atomic(count_slot).fetch_add(1u);
                            $if (res < empty_count) {
                                auto dst = index.read(res);
                                $if (dst < active_count) {
                                    relocate_frame(tok, dst);
                                };
                            };
                        };
                    }
                } else {
                    $if (src >= scan_count) { $return(); };
                    auto tok = read_scheduler_token(src, frame_buf, frame_capacity);
                    $if ((tok != 0u) & (tok < queue_count)) {
                        auto dst = def(0u);
                        auto found_dst = def(false);
                        $while (!found_dst) {
                            auto res = global.atomic(count_slot).fetch_add(1u);
                            $if (res >= empty_count) { $break; };
                            dst = index.read(res);
                            found_dst = dst < active_count;
                        };
                        $if (found_dst) {
                            relocate_frame(tok, dst);
                        };
                    };
                }
            };
        _compact_shader = _compile_shader(
            device, compact_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_compact"),
            "wavefront_compact");

        if (std::getenv("LUISA_CORO_SHADER_MAP") != nullptr) {
            for (auto &&info : _shader_infos) {
                LUISA_INFO(
                    "Wavefront shader map: stage='{}' "
                    "structural_hash={:016x}.",
                    info.stage, info.structural_hash);
            }
        }
    }

    template<typename Prepare>
    void _claim_extension_stages(
        Stream &stream, Prepare &prepare) noexcept {
        auto context = WavefrontCoroExtensionPrepareContext{
            .device = *_device,
            .stream = stream,
            .frame_desc = _extension_frame_desc,
            .frame_layout = _frame_layout,
            .frame_capacity = _config.thread_count,
            .global_memory_soa = _config.global_memory_soa,
            .shader_option = _config.shader_option};
        for (size_t i = 0u; i < _extension_stages.size(); ++i) {
            auto &registered = _extension_stages[i];
            if (registered.handler != nullptr) { continue; }
            luisa::unique_ptr<ExtensionHandler> handler{
                prepare(context, std::as_const(registered.stage))};
            if (handler == nullptr) { continue; }
            LUISA_ASSERT(!handler->name().empty(),
                         "Wavefront Extension handler name must be "
                         "non-empty for Extension '{}' at queue {}.",
                         registered.stage.extension->schema(),
                         registered.stage.queue_index);
            _last_dispatch_stats.extensions[i].handler =
                luisa::string{handler->name()};
            registered.handler = std::move(handler);
        }
    }

    void _coalesce_resume_annotation_queues() noexcept {
        using BatchEntry = detail::WavefrontCoroResumeBatchEntry;
        luisa::vector<luisa::vector<BatchEntry>> batches(_extension_stages.size());
        luisa::vector<uint> canonical(_queue_count);
        for (auto i = 0u; i < canonical.size(); ++i) { canonical[i] = i; }
        for (size_t i = 0u; i < _extension_stages.size(); ++i) {
            auto &registered = _extension_stages[i];
            if (registered.before_resume_chain.empty()) { continue; }
            auto &batch = batches[i];
            for (auto index : registered.before_resume_chain) {
                auto &member = _extension_stages[index];
                batch.emplace_back(BatchEntry{&member.stage, member.handler.get()});
            }
            for (size_t j = 0u; j < i; ++j) {
                auto &representative = _extension_stages[j];
                auto queue = representative.stage.queue_index;
                if (batches[j].empty() || canonical[queue] != queue ||
                    !detail::wavefront_coro_resume_batch_compatible(
                        luisa::span<const BatchEntry>{batch},
                        luisa::span<const BatchEntry>{batches[j]})) { continue; }
                canonical[registered.stage.queue_index] = static_cast<uint>(queue);
                if (_config.report_stats || std::getenv("LUISA_CORO_WAVEFRONT_STATS") != nullptr) {
                    LUISA_INFO(
                        "Wavefront before-resume batch alias: queue={} boundary={} "
                        "-> queue={} boundary={} continuation={}.",
                        registered.stage.queue_index, registered.stage.boundary->index,
                        queue, representative.stage.boundary->index,
                        registered.stage.boundary->to_index);
                }
                break;
            }
        }
        // Shader-side source routes read this dynamic table. Redirect only
        // suffix heads after all independent semantic prefixes are connected;
        // no source spill, compiler boundary or binding descriptor is changed.
        // Equal resident certificates make the representative queue's existing
        // compaction plan valid for the disjoint union of incoming frames.
        for (auto &queue : _extension_route_table) { queue = canonical[queue]; }
        for (auto &registered : _extension_stages) {
            registered.next_queue = canonical[registered.next_queue];
        }
        // Retain every prepared handler, including inactive aliases: commands
        // enqueued during preparation can still reference its owned resources.
    }

    void _finalize_extension_handlers() noexcept {
        if (_extension_handlers_finalized) { return; }
        _queue_deferred.assign(_queue_count, 0u);
        _queue_deferred_self_loop.assign(_queue_count, 0u);
        _queue_deferred_target.assign(_queue_count, 0u);
        _queue_blocks_deferred.assign(_queue_count, 0u);
        if (_extension_stages.empty()) {
            _extension_route_table.clear();
            _extension_handlers_finalized = true;
            return;
        }
        auto route_count = std::max<size_t>(
            _extension_boundaries.size(), 1u);
        _extension_route_table.assign(route_count, 0u);
        for (auto &&boundary : _extension_boundaries) {
            luisa::vector<size_t> active;
            for (size_t i = 0u; i < _extension_stages.size(); ++i) {
                auto &registered = _extension_stages[i];
                if (registered.stage.boundary != &boundary) { continue; }
                auto *extension = registered.stage.extension;
                if (registered.handler != nullptr) {
                    active.emplace_back(i);
                    continue;
                }
                LUISA_ASSERT(extension != nullptr,
                             "Coroutine Extension stage is null.");
                LUISA_ASSERT(
                    extension->is_annotation(),
                    "Wavefront scheduler has no handler for semantic "
                    "Extension '{}' v{} at boundary {}. Register a matching "
                    "WavefrontCoroSchedulerExtensionHandler before dispatch.",
                    extension->schema(), extension->version(),
                    boundary.index);
                LUISA_ASSERT(
                    registered.stage.dataflow->def.slots.empty(),
                    "Unhandled coroutine annotation '{}' at boundary {} "
                    "writes frame state and cannot be skipped safely.",
                    extension->schema(), boundary.index);
                switch (extension->fallback()) {
                    case CoroSuspendFallback::ignore: break;
                    case CoroSuspendFallback::warn:
                        LUISA_WARNING(
                            "Wavefront scheduler is ignoring unhandled "
                            "coroutine annotation '{}' v{} at boundary {}.",
                            extension->schema(), extension->version(),
                            boundary.index);
                        break;
                    case CoroSuspendFallback::reject:
                        LUISA_ERROR_WITH_LOCATION(
                            "Wavefront scheduler has no handler for required "
                            "coroutine annotation '{}' v{} at boundary {}.",
                            extension->schema(), extension->version(),
                            boundary.index);
                }
            }
            if (active.empty()) { continue; }
            _extension_route_table[boundary.index] = static_cast<uint>(
                _extension_stages[active.front()].stage.queue_index);
            for (size_t i = 0u; i < active.size(); ++i) {
                auto next = i + 1u < active.size() ?
                                static_cast<uint>(
                                    _extension_stages[active[i + 1u]]
                                        .stage.queue_index) :
                                static_cast<uint>(boundary.to_index);
                _extension_stages[active[i]].next_queue = next;
            }
            auto suffix = active.size();
            for (size_t i = 0u; i < active.size(); ++i) {
                auto &registered = _extension_stages[active[i]];
                if (registered.handler->execution() ==
                    WavefrontCoroExtensionExecution::before_resume) {
                    LUISA_ASSERT(boundary.to_index != 0u &&
                                     registered.stage.extension->is_annotation() &&
                                     registered.stage.dataflow->def.slots.empty() &&
                                     std::all_of(registered.stage.extension->bindings().begin(),
                                                 registered.stage.extension->bindings().end(),
                                                 [](auto &&binding) noexcept {
                                                     return binding.access == CoroSuspendBindingAccess::read;
                                                 }),
                                 "Before-resume handler '{}' must be a read-only annotation.",
                                 registered.handler->name());
                    suffix = std::min(suffix, i);
                } else {
                    LUISA_ASSERT(suffix == active.size(),
                                 "Independent Extension '{}' follows a before-resume annotation "
                                 "at boundary {}; before-resume handlers must form a suffix.",
                                 registered.handler->name(), boundary.index);
                    if (registered.handler->defer_while_other_queues_nonempty() &&
                        registered.stage.queue_index < _queue_deferred.size()) {
                        const auto q = registered.stage.queue_index;
                        _queue_deferred[q] = 1u;
                        _queue_deferred_target[q] = static_cast<uint>(boundary.to_index);
                        if (boundary.from_index == boundary.to_index) {
                            _queue_deferred_self_loop[q] = 1u;
                        }
                        if (boundary.from_index < _queue_blocks_deferred.size()) {
                            _queue_blocks_deferred[boundary.from_index] = 1u;
                        }
                        if (boundary.to_index < _queue_blocks_deferred.size()) {
                            _queue_blocks_deferred[boundary.to_index] = 1u;
                        }
                    }
                }
            }
            if (suffix != active.size()) {
                auto &first = _extension_stages[active[suffix]];
                first.before_resume_chain.assign(active.data() + suffix, active.data() + active.size());
                auto queue = first.stage.queue_index;
                _queue_continuation[queue] = static_cast<uint>(boundary.to_index);
                _refill_at[queue] = _refill_at[boundary.to_index];
            }
        }
        _coalesce_resume_annotation_queues();
        auto owner_of = [&](size_t idx) noexcept -> size_t {
            return idx < _queue_continuation.size() &&
                           _queue_continuation[idx] < _continuation_count ?
                       static_cast<size_t>(_queue_continuation[idx]) :
                       idx;
        };
        for (auto &registered : _extension_stages) {
            const auto q = registered.stage.queue_index;
            if (q < _queue_deferred.size() && _queue_deferred[q] != 0u &&
                registered.next_queue != q &&
                registered.next_queue < _queue_blocks_deferred.size()) {
                _queue_blocks_deferred[registered.next_queue] = 1u;
                auto next_owner = owner_of(registered.next_queue);
                if (next_owner < _queue_blocks_deferred.size()) {
                    _queue_blocks_deferred[next_owner] = 1u;
                }
            }
        }
        for (size_t i = 0u; i < _queue_count; ++i) {
            auto owner = owner_of(i);
            if (owner < _queue_blocks_deferred.size() &&
                _queue_blocks_deferred[owner] != 0u) {
                _queue_blocks_deferred[i] = 1u;
            }
        }
        _extension_handlers_finalized = true;
    }

    void _encode_sort_token_buckets(CommandList &commands, Stream &stream, uint count) noexcept {
        if (count == 0u) {
            if (!commands.empty()) { stream << commands.commit(); }
            stream.synchronize();
            std::fill(_host_count.begin(), _host_count.end(), 0u);
            std::fill(_host_offset.begin(), _host_offset.end(), 0u);
            return;
        }
        _sort_token.encode_sort(commands, _sort_key[0].view(), _resume_index.view(),
                                _sort_key[1].view(), _resume_index.view(),
                                count, _frame_buffer, _config.thread_count);
        commands << _sort_temp_storage.hist_buffer.view()
                        .subview(0u, _host_offset.size())
                        .copy_to(luisa::span{_host_offset.data(), _host_offset.size()});
        stream << commands.commit();
        stream.synchronize();
        for (auto i = 0u; i < _host_count.size(); i++) {
            auto next = i + 1u == _host_count.size() ? count : _host_offset[i + 1u];
            _host_count[i] = next - _host_offset[i];
        }
    }

    [[nodiscard]] auto _encode_sort_hint_range(CommandList &commands, uint offset, uint count) noexcept {
        BufferView<uint> indices[2] = {
            _resume_index.view().subview(offset, count),
            _sort_index.view().subview(offset, count)};
        BufferView<uint> keys[2] = {
            _sort_key[0].view().subview(offset, count),
            _sort_key[1].view().subview(offset, count)};
        return _sort_hint.encode_sort_switch(commands, keys, indices, count,
                                             _resume_index.view().subview(offset, count),
                                             _frame_buffer, _config.thread_count,
                                             _config.hint_range,
                                             _config.hint_partition_size);
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        _finalize_extension_handlers();
        auto pending_commands = CommandList::create(16u);
        auto flush_pending = [&]() noexcept {
            if (!pending_commands.empty()) {
                stream << pending_commands.commit();
            }
        };
        auto sync_pending = [&]() noexcept {
            flush_pending();
            stream.synchronize();
        };
        if (!_extension_route_table.empty()) {
            pending_commands << _extension_route_buffer.copy_from(
                luisa::span{_extension_route_table});
        }
        uint N = dispatch_size.x * dispatch_size.y * dispatch_size.z;
        auto report_stats = _config.report_stats ||
                            std::getenv("LUISA_CORO_WAVEFRONT_STATS") != nullptr;
        _last_dispatch_stats.collected = report_stats;
        _last_dispatch_stats.iteration_count = 0u;
        _last_dispatch_stats.generated_count = 0u;
        _last_dispatch_stats.resumed_count = 0u;
        _last_dispatch_stats.extension_count = 0u;
        _last_dispatch_stats.gather_scan_count = 0u;
        _last_dispatch_stats.compact_scan_count = 0u;
        _last_dispatch_stats.max_scan_count = 0u;
        _last_dispatch_stats.max_active_count = 0u;
        _last_dispatch_stats.elapsed_ms = 0.0;
        for (auto &continuation : _last_dispatch_stats.continuations) {
            continuation.dispatch_count = 0u;
            continuation.executed_count = 0u;
            continuation.peak_queued_count = 0u;
        }
        for (auto &extension : _last_dispatch_stats.extensions) {
            extension.dispatch_count = 0u;
            extension.executed_count = 0u;
            extension.peak_queued_count = 0u;
        }
        for (auto &work : _last_dispatch_stats.auxiliary_work) {
            work.dispatch_count = 0u;
            work.executed_count = 0u;
            work.peak_queued_count = 0u;
        }
        // `thread_count` is the allocated frame-pool ceiling, not a mandate to
        // initialize and scan every slot. A paper-scale pool (2^24 frames) is
        // intentionally much larger than many tiled or diagnostic dispatches;
        // only logical instances that can exist in this dispatch are active.
        _active_frame_capacity = std::min(_config.thread_count, N);
        if (_active_frame_capacity == 0u) {
            flush_pending();
            return;
        }
        pending_commands << _initialize_shader(
                                _config.thread_count,
                                _active_frame_capacity,
                                _resume_index)
                                .dispatch(_active_frame_capacity);
        _used_frame_count = 0u;

        auto nc = _resume_kernels.size();
        auto nq = static_cast<size_t>(_queue_count);
        for (size_t i = 0u; i < nq; ++i) {
            _host_count[i] =
                _config.incremental_continuation_counts ?
                    0u :
                    (i == 0u ? _active_frame_capacity : 0u);
            _host_offset[i] = i == 0u ? 0u : _active_frame_capacity;
        }
        if (_config.incremental_continuation_counts) {
            pending_commands << _clear_count_shader(
                                    _resume_count, static_cast<uint>(nq))
                                    .dispatch(static_cast<uint>(nq));
        } else {
            pending_commands << _resume_count.copy_from(
                luisa::span{_host_count.data(), _host_count.size()});
        }
        pending_commands << _clear_count_shader(_global_buffer, 2u).dispatch(2u);
        for (auto &&registered : _auxiliary_work) {
            registered.work->encode_reset(stream, pending_commands);
        }
        _has_dispatched = true;

        auto dispatch_counter = 0u;
        auto global_count_slot = 0u;
        auto trace_iterations =
            std::getenv("LUISA_CORO_WAVEFRONT_TRACE_ITERATIONS") != nullptr;
        auto verify_queues =
            std::getenv("LUISA_CORO_WAVEFRONT_VERIFY_QUEUES") != nullptr;
        auto counts_known_on_host =
            _config.incremental_continuation_counts && !verify_queues;
        luisa::vector<bool> deterministic_sink_continuation(nc, false);
        if (_config.incremental_continuation_counts &&
            _config.fused_continuation_counts && !verify_queues) {
            for (size_t c = 1u; c < nc; ++c) {
                if (!_is_terminal_continuation[c]) { continue; }
                bool emits_aux = false;
                for (auto &&registered : _auxiliary_work) {
                    if (registered.max_emitted_per_continuation[c] != 0u) {
                        emits_aux = true;
                        break;
                    }
                }
                deterministic_sink_continuation[c] = !emits_aux;
            }
        }
        bool entry_emits_aux = false;
        for (auto &&registered : _auxiliary_work) {
            if (!registered.max_emitted_per_continuation.empty() &&
                registered.max_emitted_per_continuation[0u] != 0u) {
                entry_emits_aux = true;
                break;
            }
        }
        auto iteration_count = uint64_t{0u};
        auto gather_scan_count = uint64_t{0u};
        auto compact_scan_count = uint64_t{0u};
        auto generated_count = uint64_t{0u};
        auto resumed_count = uint64_t{0u};
        auto max_scan_count = 0u;
        auto max_active_count = 0u;
        luisa::vector<uint> logical_count(nq, 0u);
        luisa::vector<uint> logical_members(nq, 0u);
        luisa::vector<uint> deferred_target_count(nc, 0u);
        auto last_deferred_target = nc;
        auto queue_owner = [&](size_t queue) noexcept {
            // Independent Extensions have distinct scheduling identities;
            // only before-resume suffixes belong to their continuation.
            return _queue_continuation[queue] < nc ?
                       _queue_continuation[queue] : static_cast<uint>(queue);
        };
        auto resume = [&](size_t continuation, uint source_queue,
                          BufferView<uint> indices, uint count) noexcept {
            pending_commands << _resume_kernels[continuation](
                                    _config.thread_count, indices, 0u, count, source_queue, args...)
                                    .dispatch(count);
            if (_config.incremental_continuation_counts &&
                !_config.fused_continuation_counts) {
                pending_commands << _publish_resumed_count_shader(
                                        _config.thread_count, indices, 0u, _resume_count,
                                        source_queue, count)
                                        .dispatch(count);
            }
            resumed_count += count;
            if (report_stats) {
                auto &stat = _last_dispatch_stats.continuations[continuation];
                stat.dispatch_count++;
                stat.executed_count += count;
            }
        };
        Clock dispatch_clock;
        while (true) {
            iteration_count++;
            auto scan_count = _config.frame_buffer_compaction ? _used_frame_count : _active_frame_capacity;
            // A non-draining queue may execute indefinitely. Keep explicit
            // diagnostics useful without turning the failure mode into an
            // unbounded log producer: retain the prefix and then sample at
            // powers of two.
            auto trace_this_iteration =
                trace_iterations &&
                (iteration_count <= 64u ||
                 (iteration_count & (iteration_count - 1u)) == 0u);
            if (trace_this_iteration) {
                LUISA_INFO(
                    "Wavefront iteration {} begin: dispatched={} used={} scan={} capacity={}.",
                    iteration_count, dispatch_counter, _used_frame_count,
                    scan_count, _active_frame_capacity);
            }
            // The incremental policy defers its one selected-token scan until
            // after host selection. Legacy policies materialize every token
            // bucket here.
            if (!_config.incremental_continuation_counts) {
                gather_scan_count += scan_count;
            }
            max_scan_count = std::max(max_scan_count, scan_count);
            // Queue-count readbacks are ordered after all producers and side
            // consumers from the previous iteration. The scheduler's normal
            // main-queue synchronization below also completes these copies,
            // so observing side queues adds no host synchronization point.
            auto skip_readback =
                _config.incremental_continuation_counts &&
                counts_known_on_host && !verify_queues;
            if (!skip_readback) {
                for (auto &&registered : _auxiliary_work) {
                    registered.work->encode_count_readback(stream, pending_commands);
                }
            }
            if (_config.incremental_continuation_counts) {
                if (!skip_readback) {
                    pending_commands << _resume_count.copy_to(
                        luisa::span{_host_count.data(),
                                    _host_count.size()});
                    sync_pending();
                }
                counts_known_on_host = false;
                if (verify_queues && scan_count != 0u) {
                    pending_commands << _clear_count_shader(
                                            _resume_offset, static_cast<uint>(nq))
                                            .dispatch(static_cast<uint>(nq));
                    pending_commands << _count_shader(
                                            _config.thread_count,
                                            _resume_offset, scan_count)
                                            .dispatch(scan_count);
                    luisa::vector<uint> actual(nq);
                    pending_commands << _resume_offset.copy_to(luisa::span{actual});
                    sync_pending();
                    for (size_t i = 1u; i < nq; ++i) {
                        LUISA_ASSERT(
                            actual[i] == _host_count[i],
                            "Incremental wavefront queue invariant violation "
                            "at iteration {}, continuation {}: maintained "
                            "count {} differs from materialized count {}.",
                            iteration_count, i, _host_count[i], actual[i]);
                    }
                }
            } else if (_config.gather_by_sorting) {
                _encode_sort_token_buckets(pending_commands, stream, scan_count);
                if (verify_queues && scan_count != 0u) {
                    luisa::vector<uint> indices(scan_count);
                    pending_commands << _resume_index.view().subview(0u, scan_count)
                                            .copy_to(luisa::span{indices});
                    sync_pending();
                    luisa::vector<uint> seen(scan_count, 0u);
                    for (auto slot = 0u; slot < scan_count; slot++) {
                        auto index = indices[slot];
                        LUISA_ASSERT(
                            index < scan_count,
                            "Wavefront sorted queue invariant violation at iteration {}, slot {}: "
                            "frame index {} exceeds scanned prefix {}.",
                            iteration_count, slot, index, scan_count);
                        LUISA_ASSERT(
                            seen[index] == 0u,
                            "Wavefront sorted queue invariant violation at iteration {}, slot {}: "
                            "frame index {} appears more than once.",
                            iteration_count, slot, index);
                        seen[index] = 1u;
                    }
                }
            } else {
                pending_commands << _clear_count_shader(
                                        _resume_count, static_cast<uint>(nq))
                                        .dispatch(static_cast<uint>(nq));
                if (scan_count != 0u) {
                    pending_commands << _count_shader(
                                            _config.thread_count,
                                            _resume_count, scan_count)
                                            .dispatch(scan_count);
                }
                pending_commands << _resume_count.copy_to(luisa::span{_host_count.data(), _host_count.size()});
                sync_pending();
            }

            auto active_count = 0u;
            std::fill(logical_count.begin(), logical_count.end(), 0u);
            std::fill(logical_members.begin(), logical_members.end(), 0u);
            for (size_t i = 1u; i < nq; ++i) {
                active_count += _host_count[i];
                auto owner = queue_owner(i);
                logical_count[owner] += _host_count[i];
                logical_members[owner] += static_cast<uint>(_host_count[i] != 0u);
                if (report_stats && i >= nc) {
                    auto &extension = _last_dispatch_stats.extensions[i - nc];
                    extension.peak_queued_count = std::max(
                        extension.peak_queued_count, _host_count[i]);
                }
            }
            if (report_stats) {
                for (size_t i = 1u; i < nc; ++i) {
                    auto &continuation = _last_dispatch_stats.continuations[i];
                    continuation.peak_queued_count = std::max(
                        continuation.peak_queued_count, logical_count[i]);
                }
            }
            max_active_count = std::max(max_active_count, active_count);
            auto deferred_count = 0u;
            auto deferred_blocker_nonempty = false;
            for (size_t i = 1u; i < nq; ++i) {
                if (i < _queue_deferred.size() && _queue_deferred[i] != 0u) {
                    deferred_count += _host_count[i];
                } else if (_host_count[i] != 0u &&
                           i < _queue_blocks_deferred.size() &&
                           _queue_blocks_deferred[i] != 0u) {
                    deferred_blocker_nonempty = true;
                }
            }
            auto auxiliary_active_count = uint64_t{0u};
            auto selected_auxiliary = _auxiliary_work.size();
            auto selected_auxiliary_stage = 0u;
            auto selected_auxiliary_count = 0u;
            for (size_t i = 0u; i < _auxiliary_work.size(); ++i) {
                auto &&registered = _auxiliary_work[i];
                auto count = registered.work->host_count();
                LUISA_ASSERT(
                    count <= registered.work->capacity(),
                    "Wavefront auxiliary queue '{}' contains {} items, "
                    "exceeding its capacity {}.",
                    registered.work->name(), count,
                    registered.work->capacity());
                auxiliary_active_count += count;
                LUISA_ASSERT(registered.stage_count == registered.work->stage_count(),
                             "Wavefront auxiliary stage count changed after registration.");
                auto stage_total = uint64_t{0u};
                for (auto stage = 0u; stage < registered.stage_count; ++stage) {
                    auto stage_count = registered.work->stage_host_count(stage);
                    stage_total += stage_count;
                    if (report_stats) {
                        auto &work = _last_dispatch_stats.auxiliary_work[
                            registered.stats_offset + stage];
                        work.peak_queued_count = std::max(
                            work.peak_queued_count, stage_count);
                    }
                    if (stage_count > selected_auxiliary_count) {
                        selected_auxiliary = i;
                        selected_auxiliary_stage = stage;
                        selected_auxiliary_count = stage_count;
                    }
                }
                LUISA_ASSERT(stage_total == count,
                             "Wavefront auxiliary '{}' stage sum {} differs from live count {}.",
                             registered.work->name(), stage_total, count);
            }
            LUISA_ASSERT(active_count <= scan_count,
                         "Wavefront coroutine queue invariant violation: active frames ({}) exceed scanned frame prefix ({}).",
                         active_count, scan_count);
            // Incremental accounting deliberately omits queue zero. Its
            // cardinality follows from the fixed-slot conservation law
            //
            //   free = capacity - sum(C[t], t != 0).
            //
            // Legacy non-compacting scheduling still materializes queue zero
            // and therefore uses that observed cardinality directly.
            auto empty_count =
                _config.frame_buffer_compaction ||
                        _config.incremental_continuation_counts ?
                    _active_frame_capacity - active_count :
                    _host_count[0u];
            auto compact_empty_count = _config.frame_buffer_compaction ?
                                           scan_count - active_count :
                                           empty_count;
            _host_count[0u] = compact_empty_count;
            if (trace_this_iteration) {
                LUISA_INFO(
                    "Wavefront iteration {} gathered: active={} empty={} compact_empty={}.",
                    iteration_count, active_count, empty_count,
                    compact_empty_count);
            }

            auto active_offset = 0u;
            for (size_t i = 0u; i < nq; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
            }
            if (dispatch_counter == N && active_count == 0u &&
                auxiliary_active_count == 0u) {
                if (report_stats) {
                    sync_pending();
                } else {
                    flush_pending();
                }
                break;
            }

            auto selected = nq;
            auto selected_count = 0u;
            auto selected_priority = nq;
            auto select_queue = [&](bool allow_deferred) noexcept {
                selected = nq;
                selected_count = 0u;
                selected_priority = nq;
                for (size_t i = 1u; i < nq; ++i) {
                    auto is_deferred =
                        i < _queue_deferred.size() && _queue_deferred[i] != 0u;
                    if (is_deferred != allow_deferred) { continue; }
                    // Compare the complete continuation, not its separately
                    // annotated entry queues. Ascending owners retain tie order.
                    if (logical_count[i] > selected_count) {
                        selected_count = logical_count[i];
                        selected_priority = i;
                    }
                }
                for (size_t i = 1u; i < nq; ++i) {
                    if (_host_count[i] != 0u && queue_owner(i) == selected_priority) {
                        selected = i;
                        break;
                    }
                }
            };
            auto select_blocker_queue = [&]() noexcept {
                selected = nq;
                selected_count = 0u;
                selected_priority = nq;
                for (size_t i = 1u; i < nq; ++i) {
                    if (i < _queue_deferred.size() && _queue_deferred[i] != 0u) { continue; }
                    if (i >= _queue_blocks_deferred.size() || _queue_blocks_deferred[i] == 0u) { continue; }
                    if (logical_count[i] > selected_count) {
                        selected_count = logical_count[i];
                        selected_priority = i;
                    }
                }
                for (size_t i = 1u; i < nq; ++i) {
                    if (_host_count[i] != 0u && queue_owner(i) == selected_priority) {
                        selected = i;
                        break;
                    }
                }
            };
            auto select_deferred_queue = [&]() noexcept {
                selected = nq;
                selected_count = 0u;
                selected_priority = nq;
                std::fill(deferred_target_count.begin(), deferred_target_count.end(), 0u);
                for (size_t i = nc; i < nq; ++i) {
                    if (i < _queue_deferred.size() && _queue_deferred[i] != 0u) {
                        const auto target = _queue_deferred_target[i];
                        if (target < nc) {
                            deferred_target_count[target] += _host_count[i];
                        }
                    }
                }
                auto best_target = nc;
                auto best_target_count = 0u;
                for (size_t t = 0u; t < nc; ++t) {
                    if (deferred_target_count[t] > best_target_count) {
                        best_target_count = deferred_target_count[t];
                        best_target = t;
                    }
                }
                if (best_target_count == 0u) { return; }
                const auto last_target_count =
                    last_deferred_target < nc ?
                        deferred_target_count[last_deferred_target] :
                        0u;
                const auto chosen_target =
                    (last_deferred_target < nc && last_target_count != 0u &&
                     static_cast<uint64_t>(last_target_count) * 8ull >=
                         static_cast<uint64_t>(best_target_count)) ?
                        last_deferred_target :
                        best_target;
                auto best_entry = nq;
                auto best_entry_count = 0u;
                auto best_self = nq;
                auto best_self_count = 0u;
                for (size_t i = nc; i < nq; ++i) {
                    if (i >= _queue_deferred.size() || _queue_deferred[i] == 0u ||
                        _queue_deferred_target[i] != chosen_target ||
                        _host_count[i] == 0u) {
                        continue;
                    }
                    const auto c = _host_count[i];
                    if (i < _queue_deferred_self_loop.size() &&
                        _queue_deferred_self_loop[i] != 0u) {
                        if (c > best_self_count) {
                            best_self_count = c;
                            best_self = i;
                        }
                    } else if (c > best_entry_count) {
                        best_entry_count = c;
                        best_entry = i;
                    }
                }
                selected = best_entry != nq ? best_entry : best_self;
                if (selected < nq) {
                    selected_count = _host_count[selected];
                    selected_priority = selected;
                }
            };
            select_queue(false);
            auto selected_is_deferred = false;
            auto selected_blocker_override = false;
            const auto deferred_threshold =
                std::min(deferred_count / 4u, std::max(_active_frame_capacity / 8u, 1u));
            if (_config.largest_continuation_first &&
                deferred_count != 0u &&
                std::max(selected_count, selected_auxiliary_count) < deferred_threshold) {
                if (deferred_blocker_nonempty) {
                    select_blocker_queue();
                    selected_blocker_override = selected_count != 0u;
                } else {
                    select_deferred_queue();
                    selected_is_deferred = selected_count != 0u;
                }
            } else if (selected_count == 0u && selected_auxiliary_count == 0u) {
                select_deferred_queue();
                selected_is_deferred = selected_count != 0u;
            }
            auto selected_continuation = selected_priority < nc ? selected_priority : nc;

            // Individual side stages compete with main continuations by
            // cardinality, but admission accounts for ALL live stages sharing
            // a capacity. Only if the greedy winner is a main producer do we
            // substitute its capacity owner's requested drain stage.
            auto forced_auxiliary = _auxiliary_work.size();
            auto forced_auxiliary_stage = 0u;
            auto forced_auxiliary_count = 0u;
            if (selected_continuation < nc && selected_count != 0u &&
                (selected_blocker_override || selected_auxiliary_count <= selected_count)) {
                for (size_t i = 0u; i < _auxiliary_work.size(); ++i) {
                    auto &&registered = _auxiliary_work[i];
                    auto bound =
                        registered.max_emitted_per_continuation[selected_continuation];
                    if (bound == 0u) { continue; }
                    auto queued = registered.work->host_count();
                    auto capacity = registered.work->capacity();
                    registered.work->encode_prepare_for_admission(stream, pending_commands);
                    auto available = registered.work->host_available_slots();
                    LUISA_ASSERT(registered.work->host_count() == queued &&
                                     registered.work->capacity() == capacity &&
                                     available <= capacity - queued &&
                                     (queued != 0u || available == capacity),
                                 "Auxiliary '{}' admission preparation changed occupancy/capacity "
                                 "or supplied invalid materializable storage.", registered.work->name());
                    auto required =
                        static_cast<uint64_t>(selected_count) * bound;
                    if (!wavefront_auxiliary_queue_can_admit(
                            capacity, queued, selected_count, bound) ||
                        required > available) {
                        LUISA_ASSERT(
                            queued != 0u,
                            "Wavefront auxiliary queue '{}' cannot admit "
                            "continuation '{}' even while empty (required={}, "
                            "capacity={}, available={}).",
                            registered.work->name(),
                            _last_dispatch_stats.continuations[selected_continuation].name,
                            required, capacity, available);
                        if (queued > forced_auxiliary_count) {
                            forced_auxiliary = i;
                            forced_auxiliary_count = queued;
                        }
                    }
                }
            }
            if (forced_auxiliary != _auxiliary_work.size()) {
                auto &&registered = _auxiliary_work[forced_auxiliary];
                forced_auxiliary_stage = registered.work->admission_stage();
                LUISA_ASSERT(
                    forced_auxiliary_stage < registered.stage_count &&
                        registered.work->stage_host_count(forced_auxiliary_stage) != 0u,
                    "Wavefront auxiliary '{}' supplied an empty admission stage.",
                    registered.work->name());
            }
            auto dispatch_auxiliary =
                forced_auxiliary != _auxiliary_work.size() ?
                    forced_auxiliary :
                    (!selected_is_deferred && !selected_blocker_override &&
                             selected_auxiliary_count > selected_count ?
                         selected_auxiliary :
                         _auxiliary_work.size());
            auto dispatch_auxiliary_stage =
                forced_auxiliary != _auxiliary_work.size() ?
                    forced_auxiliary_stage : selected_auxiliary_stage;

            // The legacy counter/gather path materializes every queue before
            // the refill decision because compaction may need queue zero's
            // empty-slot indices. Keep that established ordering unchanged;
            // only the incremental policy defers a selected-token gather.
            if (dispatch_auxiliary == _auxiliary_work.size() &&
                !_config.incremental_continuation_counts &&
                !_config.gather_by_sorting && scan_count != 0u) {
                pending_commands << _resume_offset.copy_from(
                    luisa::span{_host_offset.data(),
                                _host_offset.size()});
                pending_commands << _gather_shader(
                                        _config.thread_count,
                                        _resume_index, _resume_offset, scan_count)
                                        .dispatch(scan_count);
            }

            auto refill_threshold = _config.refill_threshold == 0u ?
                                        _active_frame_capacity / 2u :
                                        std::min(_config.refill_threshold,
                                                 _active_frame_capacity);
            auto deferred_backpressure =
                deferred_count >= std::max(refill_threshold / 2u, 1u);
            auto refill_aligned = active_count == 0u ||
                                  selected_is_deferred ||
                                  _config.refill_continuations.empty() ||
                                  (selected < _refill_at.size() &&
                                   _refill_at[selected]);
            // An empty scheduler must always admit work. In particular,
            // floor(capacity / 2) is zero for a one-frame pool, so using only
            // the threshold inequality would leave the empty state as a
            // fixed point and make forward progress impossible.
            auto should_refill =
                active_count == 0u ||
                (active_count < refill_threshold && !deferred_backpressure);
            if (dispatch_auxiliary == _auxiliary_work.size() &&
                should_refill && refill_aligned &&
                dispatch_counter < N) {
                auto gen_count = std::min(N - dispatch_counter, empty_count);
                auto frame_offset = active_count;
                auto gather_incremental_free_slots =
                    _config.incremental_continuation_counts &&
                    ((!_config.frame_buffer_compaction && empty_count != 0u &&
                      dispatch_counter != 0u) ||
                     (_config.frame_buffer_compaction && active_count != 0u &&
                      compact_empty_count != 0u));
                uint verified_hole_count = 0u;
                if (gather_incremental_free_slots) {
                    // A fixed-slot scheduler never relocates frames. Refill
                    // instead enumerates token-zero slots once and passes the
                    // resulting index queue to both generation and count
                    // publication. In compact mode only [0, frame_offset) can
                    // supply relocation destinations, and its hole count equals
                    // the active-source count in [frame_offset, scan_count).
                    auto count_slot = global_count_slot;
                    global_count_slot ^= 1u;
                    auto gather_bound = _config.frame_buffer_compaction ?
                                            frame_offset :
                                            scan_count;
                    auto gather_threads = _gather_selected_4x ?
                                              ((gather_bound + 1023u) / 1024u) * 256u :
                                              gather_bound;
                    pending_commands << _gather_selected_shader(
                                            _config.thread_count, _resume_index,
                                            _global_buffer, count_slot, 0u, gather_bound)
                                            .dispatch(gather_threads);
                    gather_scan_count += gather_bound;
                    _host_offset[0u] = 0u;
                    if (verify_queues) {
                        uint gathered_empty_count = 0u;
                        pending_commands << _global_buffer.view()
                                                .subview(count_slot, 1u)
                                                .copy_to(luisa::span{
                                                    &gathered_empty_count, 1u});
                        sync_pending();
                        verified_hole_count = gathered_empty_count;
                        if (_config.frame_buffer_compaction) {
                            auto max_holes = std::min(frame_offset, compact_empty_count);
                            LUISA_ASSERT(
                                gathered_empty_count <= max_holes,
                                "Incremental wavefront prefix hole gather "
                                "violation at iteration {}: gathered {} holes, "
                                "exceeding max {}.",
                                iteration_count, gathered_empty_count,
                                max_holes);
                        } else {
                            LUISA_ASSERT(
                                gathered_empty_count == empty_count,
                                "Incremental wavefront empty-queue gather "
                                "violation at iteration {}: gathered {} frames, "
                                "expected {}.",
                                iteration_count, gathered_empty_count,
                                empty_count);
                        }
                    }
                }
                if (_config.frame_buffer_compaction && active_count != 0u && compact_empty_count != 0u) {
                    auto count_slot = global_count_slot;
                    global_count_slot ^= 1u;
                    auto max_holes = _config.incremental_continuation_counts ?
                                         std::min(frame_offset, compact_empty_count) :
                                         compact_empty_count;
                    pending_commands << _compact_shader(_config.thread_count,
                                                        _resume_index, _global_buffer,
                                                        count_slot, frame_offset,
                                                        max_holes, scan_count)
                                            .dispatch(scan_count - frame_offset);
                    compact_scan_count += scan_count - frame_offset;
                    if (verify_queues && _config.incremental_continuation_counts) {
                        uint relocated_count = 0u;
                        pending_commands << _global_buffer.view()
                                                .subview(count_slot, 1u)
                                                .copy_to(luisa::span{
                                                    &relocated_count, 1u});
                        sync_pending();
                        LUISA_ASSERT(
                            relocated_count == verified_hole_count,
                            "Incremental wavefront compaction conservation "
                            "violation at iteration {}: relocated {} active "
                            "sources into {} prefix holes.",
                            iteration_count, relocated_count,
                            verified_hole_count);
                    }
                }
                pending_commands << _gen_kernel(_config.thread_count,
                                                _resume_index,
                                                _host_offset[0u], frame_offset,
                                                dispatch_counter, gen_count, dispatch_size, args...)
                                        .dispatch(gen_count);
                if (_config.incremental_continuation_counts &&
                    !_config.fused_continuation_counts) {
                    pending_commands << _publish_generated_count_shader(
                                            _config.thread_count, _resume_index,
                                            _host_offset[0u], _resume_count,
                                            frame_offset, gen_count)
                                            .dispatch(gen_count);
                }
                if (report_stats) {
                    auto &entry =
                        _last_dispatch_stats.continuations[0u];
                    entry.dispatch_count++;
                    entry.executed_count += gen_count;
                }
                dispatch_counter += gen_count;
                generated_count += gen_count;
                if (_config.frame_buffer_compaction) {
                    _used_frame_count = frame_offset + gen_count;
                }
                if (_deterministic_entry_target != 0u && !entry_emits_aux &&
                    _config.incremental_continuation_counts &&
                    _config.fused_continuation_counts && !verify_queues) {
                    _host_count[_deterministic_entry_target] += gen_count;
                    counts_known_on_host = true;
                }
            } else {
                if (dispatch_auxiliary != _auxiliary_work.size()) {
                    auto &&registered =
                        _auxiliary_work[dispatch_auxiliary];
                    auto count = registered.work->stage_host_count(dispatch_auxiliary_stage);
                    LUISA_ASSERT(count != 0u,
                                 "Selected an empty wavefront auxiliary "
                                 "queue '{}'.",
                                 registered.work->name());
                    registered.work->encode_dispatch_stage(dispatch_auxiliary_stage, stream, pending_commands, args...);
                    if (report_stats) {
                        auto &work = _last_dispatch_stats
                                         .auxiliary_work[registered.stats_offset + dispatch_auxiliary_stage];
                        work.dispatch_count++;
                        work.executed_count += count;
                    }
                    if (_config.incremental_continuation_counts && !verify_queues) {
                        counts_known_on_host =
                            registered.work->try_predict_stage_transition(
                                dispatch_auxiliary_stage);
                    }
                    continue;
                }
                if (selected_continuation < nc && selected_count != 0u) {
                    for (auto &&registered : _auxiliary_work) {
                        auto bound = registered.max_emitted_per_continuation[selected_continuation];
                        if (bound != 0u) {
                            // Admission has proved this product fits capacity
                            // (and therefore uint). Storage-only preparation
                            // is ordered before the producer, not before refill.
                            registered.work->encode_prepare_for_producer(
                                stream, pending_commands, static_cast<uint>(uint64_t{selected_count} * bound));
                        }
                    }
                }
                bool fused_extension_gather = false;
                bool use_identity_selected = false;
                if (_config.incremental_continuation_counts &&
                    selected < nq && selected_count != 0u &&
                    scan_count != 0u) {
                    if (logical_members[selected_priority] > 1u) {
                        // Snapshot all memberships before resuming any member.
                        // A member may requeue paths into another entry of the
                        // same continuation; those new paths must not leak
                        // into this decision's still-unconsumed snapshot.
                        pending_commands << _resume_offset.copy_from(luisa::span{_host_offset});
                        pending_commands << _gather_shader(
                                                _config.thread_count, _resume_index,
                                                _resume_offset, scan_count)
                                                .dispatch(scan_count);
                        if (verify_queues) {
                            luisa::vector<uint> ends(nq);
                            pending_commands << _resume_offset.copy_to(luisa::span{ends});
                            sync_pending();
                            for (size_t i = 0u; i < nq; ++i) {
                                LUISA_ASSERT(
                                    ends[i] >= _host_offset[i] &&
                                        ends[i] - _host_offset[i] == _host_count[i],
                                    "Logical wavefront snapshot gather violation "
                                    "at iteration {}, queue {}: expected {} frames.",
                                    iteration_count, i, _host_count[i]);
                            }
                        }
                        gather_scan_count += scan_count;
                    } else {
                        if (selected >= nc) {
                            const auto &ext_stage = _extension_stages[selected - nc];
                            if (!ext_stage.before_resume_chain.empty()) {
                                auto continuation = _queue_continuation[selected];
                                bool has_hint = _has_hint_sort &&
                                                continuation < _have_hint.size() &&
                                                _have_hint[continuation];
                                if (!has_hint) {
                                    const auto &first_handler =
                                        *_extension_stages[ext_stage.before_resume_chain.front()].handler;
                                    fused_extension_gather =
                                        first_handler.gathers_selected_frames();
                                }
                            }
                        } else if (selected_count == scan_count &&
                                   !(_has_hint_sort && selected < _have_hint.size() && _have_hint[selected])) {
                            if (_identity_initialized_count < _config.thread_count) {
                                pending_commands << _init_identity_shader(_config.thread_count)
                                                        .dispatch(_config.thread_count);
                                _identity_initialized_count = _config.thread_count;
                            }
                            use_identity_selected = true;
                        }
                        if (!fused_extension_gather && !use_identity_selected) {
                            auto count_slot = global_count_slot;
                            global_count_slot ^= 1u;
                            auto gather_threads = _gather_selected_4x ?
                                                      ((scan_count + 1023u) / 1024u) * 256u :
                                                      scan_count;
                            pending_commands << _gather_selected_shader(
                                                    _config.thread_count, _resume_index,
                                                    _global_buffer, count_slot,
                                                    static_cast<uint>(selected), scan_count)
                                                    .dispatch(gather_threads);
                            if (verify_queues) {
                                uint gathered_count = 0u;
                                pending_commands << _global_buffer.view()
                                                        .subview(count_slot, 1u)
                                                        .copy_to(luisa::span{
                                                            &gathered_count, 1u});
                                luisa::vector<uint> gathered_indices(selected_count);
                                pending_commands << _resume_index.view()
                                                        .subview(0u, selected_count)
                                                        .copy_to(luisa::span{gathered_indices});
                                sync_pending();
                                LUISA_ASSERT(
                                    gathered_count == selected_count,
                                    "Incremental wavefront selected-queue gather "
                                    "violation at iteration {}, continuation {}: "
                                    "gathered {} frames, expected {}.",
                                    iteration_count, selected, gathered_count,
                                    selected_count);
                                luisa::vector<uint8_t> seen(_config.thread_count, 0u);
                                for (auto slot = 0u; slot < selected_count; ++slot) {
                                    auto idx = gathered_indices[slot];
                                    LUISA_ASSERT(idx < _config.thread_count && seen[idx] == 0u,
                                                 "Gather duplicate/invalid frame index {} at slot {} (iter {}).",
                                                 idx, slot, iteration_count);
                                    seen[idx] = 1u;
                                }
                            }
                            gather_scan_count += scan_count;
                        }
                        _host_offset[selected] = 0u;
                    }
                }
                for (size_t i = nc; i < nq; ++i) {
                    if (_config.largest_continuation_first &&
                        queue_owner(i) != selected_priority) {
                        continue;
                    }
                    if (!_config.largest_continuation_first &&
                        i < _queue_deferred.size() &&
                        _queue_deferred[i] != 0u &&
                        active_count != deferred_count) {
                        continue;
                    }
                    auto count = _host_count[i];
                    if (count == 0u) { continue; }
                    auto stage_index = i - nc;
                    auto &registered =
                        _extension_stages[stage_index];
                    LUISA_ASSERT(
                        registered.handler != nullptr &&
                            registered.next_queue < nq,
                        "Selected unresolved coroutine Extension queue {}.",
                        i);
                    auto indices = _resume_index.view().subview(
                        _host_offset[i], count);
                    bool ungathered_head = fused_extension_gather && (i == selected);
                    auto dispatch_handler = [&](size_t index) noexcept {
                        auto &stage = _extension_stages[index];
                        CommandList *handler_commands = nullptr;
                        if (stage.handler->supports_command_list()) {
                            handler_commands = &pending_commands;
                        } else {
                            flush_pending();
                        }
                        const bool is_ungathered = ungathered_head;
                        ungathered_head = false;
                        indices = stage.handler->dispatch_queue(WavefrontCoroExtensionDispatchContext{
                            .stream = stream,
                            .frame_buffer = _frame_buffer.view(),
                            .frame_indices = indices,
                            .frame_count = count,
                            .frame_capacity = _config.thread_count,
                            .logical_dispatch_size = dispatch_size,
                            .stage = stage.stage,
                            .command_list = handler_commands,
                            .source_queue_token = static_cast<uint>(i),
                            .scan_count = scan_count,
                            .ungathered_scan = is_ungathered});
                        LUISA_ASSERT(indices.size() == count,
                                     "Extension '{}' changed selected queue cardinality.",
                                     stage.handler->name());
                        if (verify_queues && count != 0u) {
                            luisa::vector<uint> ext_indices(count);
                            pending_commands << indices.copy_to(luisa::span{ext_indices});
                            sync_pending();
                            luisa::vector<uint8_t> seen(_config.thread_count, 0u);
                            uint dup_count = 0u;
                            for (auto slot = 0u; slot < count; ++slot) {
                                auto idx = ext_indices[slot];
                                LUISA_ASSERT(idx < _config.thread_count,
                                             "Extension '{}' produced out-of-range frame index {} at slot {} (iter {}).",
                                             stage.handler->name(), idx, slot, iteration_count);
                                if (seen[idx] != 0u) { ++dup_count; }
                                seen[idx] = 1u;
                            }
                            LUISA_ASSERT(dup_count == 0u,
                                         "Extension '{}' produced {} duplicate frame indices out of {} at iteration {}!",
                                         stage.handler->name(), dup_count, count, iteration_count);
                        }
                        if (report_stats) {
                            auto &stat = _last_dispatch_stats.extensions[index];
                            stat.dispatch_count++;
                            stat.executed_count += count;
                            stat.peak_queued_count = std::max(stat.peak_queued_count, count);
                        }
                        _last_dispatch_stats.extension_count += count;
                    };
                    if (!registered.before_resume_chain.empty()) {
                        auto continuation = _queue_continuation[i];
                        if (_has_hint_sort && _have_hint[continuation]) {
                            auto sorted = _encode_sort_hint_range(pending_commands, _host_offset[i], count);
                            if (sorted != 0u) {
                                indices = _sort_index.view().subview(_host_offset[i], count);
                            }
                        }
                        for (auto index : registered.before_resume_chain) { dispatch_handler(index); }
                        resume(continuation, static_cast<uint>(i), indices, count);
                        if (_config.largest_continuation_first &&
                            logical_members[selected_priority] == 1u &&
                            continuation < deterministic_sink_continuation.size() &&
                            deterministic_sink_continuation[continuation]) {
                            _host_count[i] = 0u;
                            counts_known_on_host = true;
                        }
                        continue;
                    }
                    dispatch_handler(stage_index);
                    if (selected_is_deferred && i == selected &&
                        i < _queue_deferred_target.size()) {
                        last_deferred_target = _queue_deferred_target[i];
                    }
                    pending_commands << _advance_extension_stage_shader(
                                            _config.thread_count, indices,
                                            0u, count, static_cast<uint>(i),
                                            registered.next_queue)
                                            .dispatch(count);
                    if (_config.incremental_continuation_counts &&
                        !_config.fused_continuation_counts) {
                        pending_commands << _publish_resumed_count_shader(
                                                _config.thread_count, indices, 0u,
                                                _resume_count,
                                                static_cast<uint>(i), count)
                                                .dispatch(count);
                    }
                    if (_config.largest_continuation_first &&
                        _config.incremental_continuation_counts &&
                        _config.fused_continuation_counts && !verify_queues &&
                        logical_members[selected_priority] == 1u) {
                        _host_count[i] = 0u;
                        if (registered.next_queue != 0u &&
                            registered.next_queue < nq) {
                            _host_count[registered.next_queue] += count;
                        }
                        counts_known_on_host = true;
                    }
                }
                for (size_t i = 1u; i < nc; ++i) {
                    if (_config.largest_continuation_first && queue_owner(i) != selected_priority) {
                        continue;
                    }
                    auto count = _host_count[i];
                    if (count == 0u) { continue; }
                    if (_has_hint_sort && _have_hint[i]) {
                        auto sorted_index = _encode_sort_hint_range(pending_commands, _host_offset[i], count);
                        BufferView<uint> indices[2] = {
                            _resume_index.view().subview(_host_offset[i], count),
                            _sort_index.view().subview(_host_offset[i], count)};
                        resume(i, static_cast<uint>(i), indices[sorted_index], count);
                    } else if (use_identity_selected && i == selected) {
                        resume(i, static_cast<uint>(i),
                               _identity_index.view().subview(0u, count), count);
                    } else {
                        resume(i, static_cast<uint>(i),
                               _resume_index.view().subview(_host_offset[i], count), count);
                    }
                    if (_config.largest_continuation_first &&
                        logical_members[selected_priority] == 1u &&
                        deterministic_sink_continuation[i]) {
                        _host_count[i] = 0u;
                        counts_known_on_host = true;
                    }
                }
            }
        }
        if (report_stats) {
            _last_dispatch_stats.iteration_count = iteration_count;
            _last_dispatch_stats.generated_count = generated_count;
            _last_dispatch_stats.resumed_count = resumed_count;
            _last_dispatch_stats.gather_scan_count = gather_scan_count;
            _last_dispatch_stats.compact_scan_count = compact_scan_count;
            _last_dispatch_stats.max_scan_count = max_scan_count;
            _last_dispatch_stats.max_active_count = max_active_count;
            _last_dispatch_stats.elapsed_ms = dispatch_clock.toc();
            LUISA_INFO("Wavefront stats: iterations={} generated={} resumed={} extensions={} gather_scan={} compact_scan={} max_scan={} max_active={} elapsed_ms={:.3f}",
                       iteration_count, generated_count, resumed_count,
                       _last_dispatch_stats.extension_count,
                       gather_scan_count, compact_scan_count,
                       max_scan_count, max_active_count,
                       _last_dispatch_stats.elapsed_ms);
            for (auto &&continuation :
                 _last_dispatch_stats.continuations) {
                LUISA_INFO(
                    "Wavefront continuation: index={} token={} name='{}' "
                    "dispatches={} executed={} peak_queued={}.",
                    continuation.index, continuation.token,
                    continuation.name, continuation.dispatch_count,
                    continuation.executed_count,
                    continuation.peak_queued_count);
            }
            for (auto &&extension :
                 _last_dispatch_stats.extensions) {
                LUISA_INFO(
                    "Wavefront Extension: queue={} boundary={} extension={} "
                    "schema='{}' handler='{}' dispatches={} executed={} "
                    "peak_queued={}.",
                    extension.queue_index, extension.boundary_index,
                    extension.extension_index, extension.schema,
                    extension.handler, extension.dispatch_count,
                    extension.executed_count,
                    extension.peak_queued_count);
            }
            for (auto &&work :
                 _last_dispatch_stats.auxiliary_work) {
                LUISA_INFO(
                    "Wavefront auxiliary: name='{}' dispatches={} "
                    "executed={} peak_queued={}.",
                    work.name, work.dispatch_count,
                    work.executed_count, work.peak_queued_count);
            }
        }
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] const CoroFrameIOPlan &frame_io_plan() const noexcept {
        return _frame_io_plan;
    }
    [[nodiscard]] uint active_frame_capacity() const noexcept { return _active_frame_capacity; }
    [[nodiscard]] const WavefrontCoroDispatchStats &
    last_dispatch_stats() const noexcept {
        return _last_dispatch_stats;
    }
    /// Append one responsibility-chain facade. The facade is called once for
    /// each still-unclaimed static Extension stage and returns nullptr to
    /// decline it or a unique stage-local handler to claim it. The facade is
    /// not retained. Registration therefore performs preparation immediately,
    /// outside render/simulation dispatch timing. Initialization commands may
    /// be enqueued on stream; dispatching on another stream requires explicit
    /// caller-managed synchronization.
    template<typename Prepare>
    void register_extension_handler(
        Stream &stream, Prepare &&prepare) noexcept {
        LUISA_ASSERT(!_has_dispatched &&
                         !_extension_handlers_finalized,
                     "Wavefront Extension handlers must be registered before "
                     "the first dispatch.");
        auto &&prepare_facade = prepare;
        _claim_extension_stages(stream, prepare_facade);
    }
    void register_auxiliary_work(
        luisa::shared_ptr<AuxiliaryWork> work) noexcept {
        LUISA_ASSERT(work != nullptr,
                     "Cannot register null wavefront auxiliary work.");
        LUISA_ASSERT(!_has_dispatched,
                     "Wavefront auxiliary work must be registered before "
                     "the first dispatch.");
        LUISA_ASSERT(
            _config.largest_continuation_first,
            "Wavefront auxiliary admission control requires greedy "
            "one-continuation-at-a-time scheduling.");
        LUISA_ASSERT(work->capacity() != 0u,
                     "Wavefront auxiliary queue '{}' must have positive "
                     "capacity.",
                     work->name());
        for (auto &&registered : _auxiliary_work) {
            LUISA_ASSERT(
                registered.work->name() != work->name(),
                "Duplicate wavefront auxiliary queue name '{}'.",
                work->name());
        }
        RegisteredAuxiliaryWork registered{
            .work = std::move(work),
            .max_emitted_per_continuation =
                luisa::vector<uint>(
                    _last_dispatch_stats.continuations.size(), 0u)};
        for (auto &&producer : registered.work->producers()) {
            LUISA_ASSERT(
                producer.max_emitted_per_invocation != 0u,
                "Wavefront auxiliary producer '{}' for queue '{}' must "
                "have a positive emission bound.",
                producer.continuation, registered.work->name());
            auto node_index =
                _last_dispatch_stats.continuations.size();
            for (size_t i = 1u;
                 i < _last_dispatch_stats.continuations.size(); ++i) {
                if (_last_dispatch_stats.continuations[i].name ==
                    producer.continuation) {
                    node_index = i;
                    break;
                }
            }
            LUISA_ASSERT(
                node_index < _last_dispatch_stats.continuations.size(),
                "Wavefront auxiliary producer '{}' for queue '{}' does "
                "not name a non-entry coroutine continuation.",
                producer.continuation, registered.work->name());
            LUISA_ASSERT(
                registered.max_emitted_per_continuation[node_index] == 0u,
                "Wavefront auxiliary queue '{}' repeats producer '{}'.",
                registered.work->name(), producer.continuation);
            auto maximum_full_queue_emission =
                static_cast<uint64_t>(_config.thread_count) *
                producer.max_emitted_per_invocation;
            LUISA_ASSERT(
                wavefront_auxiliary_queue_can_admit(
                    registered.work->capacity(), 0u,
                    _config.thread_count,
                    producer.max_emitted_per_invocation),
                "Wavefront auxiliary queue '{}' capacity {} cannot admit "
                "the proven full-queue emission bound {} from '{}'.",
                registered.work->name(), registered.work->capacity(),
                maximum_full_queue_emission, producer.continuation);
            registered.max_emitted_per_continuation[node_index] =
                producer.max_emitted_per_invocation;
        }
        LUISA_ASSERT(
            !registered.work->producers().empty(),
            "Wavefront auxiliary queue '{}' has no producers.",
            registered.work->name());
        registered.stage_count = registered.work->stage_count();
        registered.stats_offset = _last_dispatch_stats.auxiliary_work.size();
        LUISA_ASSERT(registered.stage_count != 0u,
                     "Wavefront auxiliary work must expose at least one stage.");
        for (auto stage = 0u; stage < registered.stage_count; ++stage) {
            auto name = registered.work->stage_name(stage);
            LUISA_ASSERT(!name.empty(), "Wavefront auxiliary stage name must not be empty.");
            for (auto prior = 0u; prior < stage; ++prior) {
                LUISA_ASSERT(name != registered.work->stage_name(prior),
                             "Duplicate stage '{}' in wavefront auxiliary '{}'.",
                             name, registered.work->name());
            }
            _last_dispatch_stats.auxiliary_work.emplace_back(
                WavefrontCoroAuxiliaryStats{.name = luisa::string{name}});
        }
        _auxiliary_work.emplace_back(std::move(registered));
    }
    /// Structural hashes of the scheduler-owned generate, continuation, and
    /// queue-management kernels. These exclude allocation sizes and provide a
    /// direct cache-identity diagnostic for scheduler configuration changes.
    [[nodiscard]] luisa::span<const uint64_t>
    shader_structure_hashes() const noexcept {
        return {_shader_structure_hashes.data(),
                _shader_structure_hashes.size()};
    }
    /// Semantic stage labels paired with the same hashes returned by
    /// shader_structure_hashes(), in the same compilation order.
    [[nodiscard]] luisa::span<const WavefrontCoroShaderInfo>
    shader_infos() const noexcept {
        return {_shader_infos.data(), _shader_infos.size()};
    }

    WavefrontCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config}, _device{&device} {
        LUISA_ASSERT(_config.thread_count != 0u,
                     "Wavefront coroutine frame capacity must be positive.");
        LUISA_ASSERT(_config.execution_block_size >= 32u &&
                         _config.execution_block_size <= 1024u &&
                         _config.execution_block_size % 32u == 0u,
                     "Wavefront coroutine execution block size must be a "
                     "multiple of 32 in [32, 1024], but got {}.",
                     _config.execution_block_size);
        LUISA_ASSERT(
            !_config.incremental_continuation_counts ||
                _config.largest_continuation_first,
            "Incremental selected-queue scheduling requires greedy "
            "largest-continuation-first execution.");
        LUISA_ASSERT(
            !_config.fused_continuation_counts ||
                _config.incremental_continuation_counts,
            "Fused wavefront continuation counts require incremental "
            "continuation accounting.");
        _create_shader(device, coro);
    }
    WavefrontCoroScheduler(Device &device, const Coro &coro) noexcept
        : WavefrontCoroScheduler{device, coro, Config{}} {}
};

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> WavefrontCoroScheduler<Args...>;

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                       const WavefrontCoroSchedulerConfig &)
    -> WavefrontCoroScheduler<Args...>;

}// namespace luisa::compute::coro
