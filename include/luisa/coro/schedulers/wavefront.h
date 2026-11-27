//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
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
    // Maximum number of concurrent in-flight batches for blocking_async
    // Extension stages. Defaults to 1 for legacy single-buffer behavior;
    // callers or LUISA_CORO_ASYNC_MAX_SLOTS can enable bounded multi-buffering.
    uint max_async_in_flight_batches = 1u;
    // Enable lightweight online GPU dispatch timing (piggybacked on existing
    // sync points; zero extra synchronization) and time-budget overlap scheduling
    // when blocking_async extensions are present. Can be overridden via
    // LUISA_CORO_ONLINE_TIMING=0/1 and LUISA_CORO_TIME_BUDGET_SCHED=0/1.
    bool enable_online_timing = true;
    bool enable_time_budget_scheduling = true;
    // Continuations in this set act as collection hubs in greedy scheduling:
    // when non-collect producers have pending frames, drain those producers
    // before running the hub so rays batch together for a larger, more
    // coherent dispatch, unless refill is needed. Kept last for positional
    // aggregate source compatibility.
    luisa::vector<luisa::string> collect_continuations;
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
    bool online_timing_active{false};
    bool online_timing_hw_timestamps{false};
    double online_sync_bubble_ms{0.0};
    uint64_t time_budget_overlap_dispatches{0u};
    uint64_t time_budget_overlap_waits{0u};
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
        bool is_blocking_async{false};
        bool modifies_frame{false};
        uint producer_continuation{0u};
        size_t async_domain_index{0u};
    };
    struct ExtensionRoute {
        uint token{0u};
        uint boundary_index{0u};
        luisa::vector<size_t> source_store_fields;
    };
    struct QueueTimingModel {
        uint32_t samples{0u};
        uint32_t non_overlap_samples{0u};
        double w{0.0};
        double sx{0.0};
        double sy{0.0};
        double sxx{0.0};
        double sxy{0.0};
        double alpha_ms{0.0};
        double beta_ms_per_thread{0.0};
        double ema_ref_count{0.0};
        double ema_ms_per_thread{0.0};

        void observe(uint count, double t_ms, bool is_overlap) noexcept {
            if (count == 0u || !(t_ms >= 0.0) || !std::isfinite(t_ms)) {
                return;
            }
            constexpr double kLambda = 0.88;
            constexpr double kRefDecay = 0.92;
            const double x = static_cast<double>(count);
            double y = std::clamp(t_ms, 0.001, 200.0);
            if (samples >= 4u) {
                const double pred =
                    std::max(0.001, alpha_ms + beta_ms_per_thread * x);
                if (y > 8.0 * pred) {
                    y = 4.0 * pred;
                }
            }
            const double rate = y / x;
            ema_ms_per_thread = (samples == 0u)
                                    ? rate
                                    : (0.88 * ema_ms_per_thread + 0.12 * rate);
            w = kLambda * w + 1.0;
            sx = kLambda * sx + x;
            sy = kLambda * sy + y;
            sxx = kLambda * sxx + x * x;
            sxy = kLambda * sxy + x * y;
            ++samples;
            if (!is_overlap) {
                ema_ref_count = (non_overlap_samples == 0u)
                                    ? x
                                    : (0.92 * ema_ref_count + 0.08 * x);
                ++non_overlap_samples;
            } else if (non_overlap_samples == 0u) {
                ema_ref_count = (samples == 1u)
                                    ? x
                                    : (kRefDecay * ema_ref_count +
                                       (1.0 - kRefDecay) * x);
            }
            const double mean_x = sx / w;
            const double mean_y = sy / w;
            const double var_x = std::max(0.0, sxx / w - mean_x * mean_x);
            const double spread_thresh = 0.15 * mean_x;
            const double cov_xy = sxy / w - mean_x * mean_y;
            if (samples >= 4u && var_x > spread_thresh * spread_thresh &&
                var_x > 1e-9 && cov_xy > 0.0) {
                const double slope_cap = mean_y / std::max(mean_x, 1.0);
                const double slope_floor = 0.25 * slope_cap;
                beta_ms_per_thread =
                    std::clamp(cov_xy / var_x, slope_floor, slope_cap);
                const double raw_alpha =
                    std::max(0.0, mean_y - beta_ms_per_thread * mean_x);
                if (raw_alpha > 0.35 * mean_y) {
                    alpha_ms = 0.35 * mean_y;
                    beta_ms_per_thread =
                        std::max(0.0, mean_y - alpha_ms) / std::max(mean_x, 1.0);
                } else {
                    alpha_ms = raw_alpha;
                }
            } else {
                alpha_ms = std::min(0.05, 0.25 * mean_y);
                const double base_beta =
                    std::max(0.0, mean_y - alpha_ms) / std::max(mean_x, 1.0);
                if (cov_xy <= 0.0 && samples >= 4u) {
                    beta_ms_per_thread = std::clamp(
                        0.75 * ema_ms_per_thread, base_beta,
                        std::max(base_beta, 2.5 * base_beta));
                } else {
                    beta_ms_per_thread = base_beta;
                }
            }
        }

        [[nodiscard]] double predict_ms(uint count) const noexcept {
            return alpha_ms + beta_ms_per_thread * static_cast<double>(count);
        }
    };

    Config _config;
    ByteBuffer _frame_buffer;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint, uint3, Args...>
        _gen_kernel;
    luisa::vector<Shader1D<uint, Buffer<uint>, uint, uint, uint, Args...>>
        _resume_kernels;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint>
        _advance_extension_stage_shader;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint, uint, Buffer<uint>>
        _advance_extension_stage_filtered_shader;
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
    Buffer<uint> _async_slot_source_queue;
    Buffer<uint> _advance_ready_flags;
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
    luisa::vector<bool> _collect_at;
    luisa::vector<bool> _is_terminal_continuation;
    luisa::vector<luisa::vector<uint>> _continuation_targets;
    CoroFrameStorageLayout _frame_layout;
    CoroFrameIOPlan _frame_io_plan;
    // Handlers and delayed finalization outlive the source Coroutine. Own
    // their immutable descriptors, not a borrowed compiler input or another
    // frame allocation. Declare these before handlers so they die last.
    CoroFrameDesc _extension_frame_desc;
    luisa::vector<CoroGraph::Boundary> _extension_boundaries;
    luisa::vector<RegisteredExtensionStage> _extension_stages;
    luisa::vector<luisa::string> _async_domain_names;
    luisa::vector<uint> _domain_max_async_slots;
    luisa::vector<luisa::vector<ExtensionRoute>> _extension_routes;
    luisa::vector<uint> _extension_route_table;
    // A physical queue can execute an annotation suffix and then a logical
    // continuation. Ordinary independent Extension queues have no such owner.
    luisa::vector<uint> _queue_continuation;
    luisa::vector<uint8_t> _queue_deferred;
    luisa::vector<uint8_t> _queue_deferred_self_loop;
    luisa::vector<uint> _queue_deferred_target;
    luisa::vector<uint8_t> _target_has_deferred_self_loop;
    luisa::vector<uint8_t> _queue_blocks_deferred;
    luisa::vector<uint8_t> _queue_is_async_consumer;
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
    uint _async_slot_count{0u};
    uint _total_token_count{0u};
    uint _used_frame_count{0u};
    uint _active_frame_capacity{0u};
    uint _initialized_frame_capacity{0u};
    uint _identity_initialized_count{0u};
    uint _deterministic_entry_target{0u};
    bool _has_hint_sort{false};
    bool _gather_selected_4x{false};
    bool _tail_unroll_supported{false};
    bool _has_dispatched{false};
    bool _extension_handlers_finalized{false};
    bool _timeline_labels_registered{false};
    uint64_t _continuation_dispatch_seq{0ull};
    uint64_t _async_consumer_dispatch_seq{0ull};
    uint64_t _auxiliary_dispatch_seq{0ull};
    uint64_t _next_async_generation{0ull};
    uint64_t _next_async_submit_seq{0ull};
    luisa::vector<QueueTimingModel> _timing_models;
    double _t_sync_ms{0.15};
    uint32_t _sync_bubble_samples{0u};
    luisa::vector<double> _domain_async_est_ms;
    luisa::vector<uint32_t> _domain_async_samples;

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
        uint effective_async_slots = _config.max_async_in_flight_batches;
        if (const auto *raw_slots = std::getenv("LUISA_CORO_ASYNC_MAX_SLOTS")) {
            char *end = nullptr;
            const auto parsed = std::strtoul(raw_slots, &end, 10);
            if (end != raw_slots && *end == '\0' && parsed >= 1ul && parsed <= 8ul) {
                effective_async_slots = static_cast<uint>(parsed);
            }
        }
        _async_slot_count = _extension_stages.empty() ?
                                0u :
                                std::clamp<uint>(effective_async_slots, 1u, 8u);
        _total_token_count = _queue_count + _async_slot_count;
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
        _host_count.resize(_total_token_count);
        _host_offset.resize(_total_token_count);
        _have_hint.resize(_queue_count, false);
        _refill_at.resize(_queue_count, false);
        _collect_at.resize(_queue_count, false);
        _is_terminal_continuation.assign(nc, false);
        _continuation_targets.assign(nc, {});
        for (size_t i = 0u; i < nc; ++i) {
            if (i >= 1u) {
                _is_terminal_continuation[i] = coro.graph().node(i).targets.empty();
            }
            for (auto target : coro.graph().node(i).targets) {
                if (target != 0u && target < nc) {
                    auto t = static_cast<uint>(target);
                    if (std::find(_continuation_targets[i].begin(),
                                  _continuation_targets[i].end(), t) ==
                        _continuation_targets[i].end()) {
                        _continuation_targets[i].emplace_back(t);
                    }
                }
            }
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
        for (auto &name : _config.collect_continuations) {
            auto node = coro.graph().node_by_name(name);
            LUISA_ASSERT(node != nullptr && node->index != 0u &&
                             node->index < nc,
                         "Wavefront collect continuation '{}' does not name a "
                         "valid non-entry coroutine suspension.",
                         name);
            _collect_at[node->index] = true;
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
        _resume_count = device.create_buffer<uint>(_total_token_count);
        auto *resume_count_buffer = &_resume_count;
        auto fuse_continuation_counts =
            _config.incremental_continuation_counts &&
            _config.fused_continuation_counts;
        const auto backend_name = device.backend_name();
        const auto tail_unroll_supported =
            fuse_continuation_counts &&
            (backend_name == "dx" || backend_name == "vk" || backend_name == "metal");
        _tail_unroll_supported = tail_unroll_supported;
        // Incremental queue counts are scheduler-owned state. For
        // every nonterminal continuation t, C[t] is the cardinality of the
        // frames whose target token is t. Queue zero is derived from the
        // relevant slot domain minus sum(C[1..]) and is deliberately not
        // maintained.
        // The default policy binds this buffer only in scheduler-owned
        // publication kernels. The explicitly fused policy instead binds it
        // in producers so their transition atomics need no second pass.
        _resume_offset = device.create_buffer<uint>(_total_token_count);
        _global_buffer = device.create_buffer<uint>(2u);
        auto *global_buffer = &_global_buffer;
        if (!_extension_stages.empty()) {
            _extension_route_buffer = device.create_buffer<uint>(
                coro.graph().boundary_count());
        }
        if (_async_slot_count != 0u) {
            _async_slot_source_queue = device.create_buffer<uint>(_async_slot_count);
        }
        auto *extension_route_buffer = &_extension_route_buffer;
        if (use_sort) {
            _sort_index = device.create_buffer<uint>(_config.thread_count);
            _sort_key[0] = device.create_buffer<uint>(_config.thread_count);
            _sort_key[1] = device.create_buffer<uint>(_config.thread_count);
            auto max_digit = std::max<uint>(
                _total_token_count,
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
                1u, _total_token_count, 0u, 31u,
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

        luisa::vector<luisa::move_only_function<void()>> pending_compilations;
        auto enqueue_compile = [&]<typename ShaderT, typename KernelT>(
                                   ShaderT &slot, KernelT &&kernel,
                                   ShaderOption option, luisa::string stage) {
            if (_config.shader_compile_threads <= 1u) {
                slot = _compile_shader(
                    device, kernel, std::move(option), std::move(stage));
            } else {
                // Register diagnostics in graph/build order on the constructing
                // thread. Workers own the kernel wrapper and label and write only
                // their preallocated shader slot; no shared vectors are appended.
                auto structural_hash = kernel.function()->function().hash();
                _shader_structure_hashes.emplace_back(structural_hash);
                _shader_infos.emplace_back(WavefrontCoroShaderInfo{
                    .stage = stage, .structural_hash = structural_hash});
                pending_compilations.emplace_back(
                    [&device, &slot,
                     kernel = std::forward<KernelT>(kernel),
                     option = std::move(option),
                     stage = std::move(stage)]() mutable {
                        slot = detail::coro_scheduler_label_shader(
                            device.compile(kernel, std::move(option)), stage);
                    });
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
            enqueue_compile(
                _gen_kernel,
                std::move(k_gen),
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_generate"),
                "wavefront_generate/<entry>");
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            auto cont_candidates = make_candidate_queues(i);
            Kernel1D k_cont = [&coro, frame_buffer, extension_route_buffer,
                               resume_count_buffer, global_buffer,
                               tail_unroll_supported,
                               routes = _extension_routes[i],
                               layout = _frame_layout, input_fields = _frame_io_plan.input_fields[i], output_fields = _frame_io_plan.transition_output_fields[i],
                               soa = _config.global_memory_soa, i,
                               execution_block_size = _continuation_block_sizes[i],
                               read_scheduler_token, token_to_index,
                               fuse_continuation_counts,
                               has_extensions = !_extension_stages.empty(),
                               total_token_count = _total_token_count,
                               cont_candidates = std::move(cont_candidates),
                               emit_fused_resume_counts](
                                  UInt frame_capacity,
                                  BufferUInt resume_index,
                                  UInt resume_offset, UInt count, UInt source_queue,
                                  Var<Args>... k_args) noexcept {
                set_block_size(execution_block_size);
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                if (tail_unroll_supported) {
                    auto actual_count = def(count);
                    auto is_indirect_count = (count & 0x80000000u) != 0u;
                    $if (is_indirect_count) {
                        actual_count = Expr<Buffer<uint>>{*global_buffer}.read(count & 1u);
                    };
                    count = actual_count;
                }
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
                        $if ((tok != 0u) & (tok < total_token_count)) {
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
            enqueue_compile(
                _resume_kernels[i], std::move(k_cont),
                std::move(option), std::move(stage));
        }

        if (!_extension_stages.empty()) {
            auto *async_slot_source_queue_buffer = &_async_slot_source_queue;
            const auto warp_aggregate_counts =
                device.compute_warp_size() == 32u;
            Kernel1D advance_extension_stage =
                [frame_buffer, layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 read_scheduler_token, resume_count_buffer,
                 async_slot_source_queue_buffer,
                 fuse_continuation_counts,
                 warp_aggregate_counts,
                 queue_count = _queue_count,
                 total_token_count = _total_token_count,
                 async_slot_count = _async_slot_count](
                    UInt frame_capacity, BufferUInt indices,
                    UInt index_offset, UInt count,
                    UInt current_queue, UInt next_queue) noexcept {
                    set_block_size(256u);
                    if (warp_aggregate_counts) {
                        set_warp_size(32u);
                    }
                    Shared<uint> block_next{1u};
                    if (fuse_continuation_counts) {
                        $if (thread_x() == 0u) {
                            block_next[0u] = 0u;
                        };
                        sync_block();
                    }
                    auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                    auto x = dispatch_x();
                    auto is_in_range = x < count;
                    auto is_match = def(false);
                    $if (is_in_range) {
                        if (async_slot_count != 0u) {
                            auto slot_sources =
                                Expr<Buffer<uint>>{*async_slot_source_queue_buffer};
                            $if ((x == 0u) & (next_queue >= queue_count) &
                                 (next_queue < total_token_count)) {
                                slot_sources.write(
                                    next_queue - queue_count, current_queue);
                            };
                        }
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
                        is_match = (token == current_queue);
                        $if (is_match) {
                            coro_frame_write_field(
                                frame_buf, frame_index, frame_capacity,
                                layout, soa, 6u, next_queue);
                        } $else {
                            if (fuse_continuation_counts) {
                                auto resume_count =
                                    Expr<Buffer<uint>>{*resume_count_buffer};
                                $if ((token != 0u) &
                                     (token < total_token_count)) {
                                    resume_count.atomic(token).fetch_add(1u);
                                };
                            }
                        };
                    };
                    if (fuse_continuation_counts) {
                        if (warp_aggregate_counts) {
                            auto wn = warp_active_count_bits(is_match);
                            $if (warp_lane_id() == 0u) {
                                $if (wn != 0u) {
                                    block_next.atomic(0u).fetch_add(wn);
                                };
                            };
                        } else {
                            $if (is_match) {
                                block_next.atomic(0u).fetch_add(1u);
                            };
                        }
                        sync_block();
                        $if (thread_x() == 0u) {
                            auto total_next = block_next[0u];
                            $if ((total_next != 0u) & (next_queue != 0u) &
                                 (next_queue < total_token_count)) {
                                auto resume_count =
                                    Expr<Buffer<uint>>{*resume_count_buffer};
                                resume_count.atomic(next_queue)
                                    .fetch_add(total_next);
                            };
                        };
                    }
                };
            enqueue_compile(
                _advance_extension_stage_shader,
                std::move(advance_extension_stage),
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
        enqueue_compile(
            _initialize_shader, std::move(initialize_kernel),
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_initialize"),
            "wavefront_initialize");

        if (_config.incremental_continuation_counts) {
            Kernel1D init_identity_kernel = [identity_buf = &_identity_index](UInt n) {
                auto x = dispatch_x();
                $if (x < n) { Expr<Buffer<uint>>{*identity_buf}.write(x, x); };
            };
            enqueue_compile(
                _init_identity_shader, std::move(init_identity_kernel),
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_init_identity"),
                "wavefront_init_identity");
        }
        _deterministic_entry_target = _detect_deterministic_entry_target(coro);

        Kernel1D clear_count_kernel = [](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buffer.write(x, 0u); };
        };
        enqueue_compile(
            _clear_count_shader, std::move(clear_count_kernel),
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_clear_count"),
            "wavefront_clear_count");

        Kernel1D count_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, total_token_count = _total_token_count](
                UInt frame_capacity,
                BufferUInt count, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < total_token_count) {
                count.atomic(tok).fetch_add(1u);
            };
        };
        enqueue_compile(
            _count_shader, std::move(count_kernel),
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
                 total_token_count = _total_token_count](
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
                    $if ((tok != 0u) & (tok < total_token_count)) {
                        count.atomic(tok).fetch_add(1u);
                    };
                };
            enqueue_compile(
                _publish_generated_count_shader,
                std::move(publish_generated_count_kernel),
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
                 total_token_count = _total_token_count](
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
                    $if ((tok != 0u) & (tok < total_token_count)) {
                        count.atomic(tok).fetch_add(1u);
                    };
                };
            enqueue_compile(
                _publish_resumed_count_shader,
                std::move(publish_resumed_count_kernel),
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    "wavefront_publish_resumed_count"),
                "wavefront_publish_resumed_count");
        }

        Kernel1D gather_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, total_token_count = _total_token_count](
                UInt frame_capacity,
                BufferUInt index, BufferUInt offset, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < total_token_count) {
                auto slot = offset.atomic(tok).fetch_add(1u);
                index.write(slot, x);
            };
        };
        enqueue_compile(
            _gather_shader, std::move(gather_kernel),
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
                enqueue_compile(
                    _gather_selected_shader, std::move(gather_selected_kernel),
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
                enqueue_compile(
                    _gather_selected_shader, std::move(gather_selected_kernel),
                    detail::coro_scheduler_shader_option(
                        _config.shader_option, "wavefront_gather_selected"),
                    "wavefront_gather_selected");
            }
        }

        auto relocation_partition =
            coro_frame_partition_relocation_fields(
                std::move(queue_relocation_fields),
                _config.global_memory_soa ? &_frame_layout : nullptr);
        auto *async_slot_source_queue_buffer = &_async_slot_source_queue;
        Kernel1D compact_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, desc = &coro.frame(),
             async_slot_source_queue_buffer,
             queue_count = _queue_count,
             total_token_count = _total_token_count,
             async_slot_count = _async_slot_count,
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
                    // live-begin projection. Parked async batch tokens map
                    // back to their stage's relocation plan while preserving
                    // their parked token in field 6.
                    if (!common_relocation_fields.empty()) {
                        coro_frame_copy_fields(
                            frame_buf, src, dst, frame_capacity, desc,
                            layout, soa,
                            luisa::span<const size_t>{
                                common_relocation_fields});
                    }
                    auto reloc_tok = def(tok);
                    if (async_slot_count != 0u) {
                        auto slot_sources =
                            Expr<Buffer<uint>>{*async_slot_source_queue_buffer};
                        $if (tok >= queue_count) {
                            reloc_tok = slot_sources.read(tok - queue_count);
                        };
                    }
                    auto copy_residual = switch_(reloc_tok);
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
                    auto is_active_src = (tok != 0u) & (tok < total_token_count);
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
                    $if ((tok != 0u) & (tok < total_token_count)) {
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
        enqueue_compile(
            _compact_shader, std::move(compact_kernel),
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_compact"),
            "wavefront_compact");

        detail::coro_scheduler_compile_shaders(
            pending_compilations, _config.shader_compile_threads);

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
        _async_domain_names.clear();
        _domain_max_async_slots.clear();
        _queue_deferred.assign(_queue_count, 0u);
        _queue_deferred_self_loop.assign(_queue_count, 0u);
        _queue_deferred_target.assign(_queue_count, 0u);
        _target_has_deferred_self_loop.assign(_continuation_count, 0u);
        _queue_blocks_deferred.assign(_queue_count, 0u);
        _queue_is_async_consumer.assign(_queue_count, 0u);
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
                    if (registered.handler->execution() ==
                        WavefrontCoroExtensionExecution::blocking_async) {
                        registered.is_blocking_async = true;
                        registered.modifies_frame =
                            registered.stage.dataflow != nullptr &&
                            !registered.stage.dataflow->def.slots.empty();
                        registered.producer_continuation =
                            static_cast<uint>(boundary.from_index);
                        luisa::string domain_name;
                        if (!registered.handler->async_domain().empty()) {
                            domain_name = luisa::string{
                                registered.handler->async_domain()};
                        } else if (registered.stage.dataflow != nullptr &&
                                   !registered.stage.dataflow->async_domain.empty()) {
                            domain_name = registered.stage.dataflow->async_domain;
                        } else {
                            domain_name = luisa::format("__stage_{}", active[i]);
                        }
                        auto it = std::find(_async_domain_names.begin(),
                                            _async_domain_names.end(), domain_name);
                        if (it == _async_domain_names.end()) {
                            registered.async_domain_index = _async_domain_names.size();
                            _async_domain_names.emplace_back(std::move(domain_name));
                        } else {
                            registered.async_domain_index = static_cast<size_t>(
                                it - _async_domain_names.begin());
                        }
                    }
                    if (registered.handler->defer_while_other_queues_nonempty() &&
                        registered.stage.queue_index < _queue_deferred.size()) {
                        const auto q = registered.stage.queue_index;
                        _queue_deferred[q] = 1u;
                        _queue_deferred_target[q] = static_cast<uint>(boundary.to_index);
                        if (boundary.from_index == boundary.to_index) {
                            _queue_deferred_self_loop[q] = 1u;
                            if (boundary.to_index < _target_has_deferred_self_loop.size()) {
                                _target_has_deferred_self_loop[boundary.to_index] = 1u;
                            }
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
                _collect_at[queue] = _collect_at[boundary.to_index];
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
        for (auto &registered : _extension_stages) {
            if (!registered.is_blocking_async) {
                continue;
            }
            const auto from_idx = registered.stage.boundary != nullptr
                                      ? static_cast<size_t>(registered.stage.boundary->from_index)
                                      : _queue_count;
            if (from_idx > 0u && from_idx < _queue_is_async_consumer.size()) {
                _queue_is_async_consumer[from_idx] = 1u;
                const auto from_owner = owner_of(from_idx);
                if (from_owner > 0u &&
                    from_owner < _queue_is_async_consumer.size()) {
                    _queue_is_async_consumer[from_owner] = 1u;
                }
            }
            const auto to_idx = registered.stage.boundary != nullptr
                                    ? static_cast<size_t>(registered.stage.boundary->to_index)
                                    : _queue_count;
            if (to_idx < _queue_is_async_consumer.size()) {
                _queue_is_async_consumer[to_idx] = 1u;
                const auto to_owner = owner_of(to_idx);
                if (to_owner < _queue_is_async_consumer.size()) {
                    _queue_is_async_consumer[to_owner] = 1u;
                }
            }
            const auto next_q = static_cast<size_t>(registered.next_queue);
            if (next_q < _queue_is_async_consumer.size()) {
                _queue_is_async_consumer[next_q] = 1u;
                const auto next_owner = owner_of(next_q);
                if (next_owner < _queue_is_async_consumer.size()) {
                    _queue_is_async_consumer[next_owner] = 1u;
                }
            }
        }
        for (size_t i = 0u; i < _queue_count; ++i) {
            const auto owner = owner_of(i);
            if (owner < _queue_is_async_consumer.size() &&
                _queue_is_async_consumer[owner] != 0u) {
                _queue_is_async_consumer[i] = 1u;
            }
        }
        const uint slot_cap = std::max<uint>(_async_slot_count, 1u);
        _domain_max_async_slots.assign(_async_domain_names.size(), slot_cap);
        for (auto &registered : _extension_stages) {
            if (!registered.is_blocking_async || registered.handler == nullptr) {
                continue;
            }
            if (registered.async_domain_index < _domain_max_async_slots.size()) {
                const uint handler_max =
                    std::max<uint>(1u, registered.handler->max_async_in_flight_batches());
                _domain_max_async_slots[registered.async_domain_index] = std::min(
                    _domain_max_async_slots[registered.async_domain_index], handler_max);
            }
        }
        bool any_filtered_advance = false;
        for (const auto &registered : _extension_stages) {
            if (registered.handler != nullptr &&
                registered.handler->execution() !=
                    WavefrontCoroExtensionExecution::before_resume &&
                registered.handler->supports_filtered_advance()) {
                any_filtered_advance = true;
                break;
            }
        }
        if (any_filtered_advance && _device != nullptr) {
            if (!_advance_ready_flags) {
                _advance_ready_flags =
                    _device->create_buffer<uint>(_config.thread_count);
            }
            if (!_advance_extension_stage_filtered_shader) {
                auto *frame_buffer = &_frame_buffer;
                auto *resume_count_buffer = &_resume_count;
                const bool fuse_continuation_counts =
                    _config.incremental_continuation_counts &&
                    _config.fused_continuation_counts;
                const bool warp_aggregate_counts =
                    _device->compute_warp_size() == 32u;
                Kernel1D advance_extension_stage_filtered =
                    [frame_buffer, layout = _frame_layout,
                     soa = _config.global_memory_soa,
                     resume_count_buffer,
                     fuse_continuation_counts,
                     warp_aggregate_counts,
                     total_token_count = _total_token_count](
                        UInt frame_capacity, BufferUInt indices,
                        UInt index_offset, UInt count,
                        UInt current_queue, UInt next_queue,
                        UInt retry_queue, BufferUInt ready_flags) noexcept {
                        set_block_size(256u);
                        if (warp_aggregate_counts) {
                            set_warp_size(32u);
                        }
                        Shared<uint> block_counts{2u};
                        if (fuse_continuation_counts) {
                            $if (thread_x() == 0u) {
                                block_counts[0u] = 0u;
                                block_counts[1u] = 0u;
                            };
                            sync_block();
                        }
                        auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                        auto x = dispatch_x();
                        auto is_in_range = x < count;
                        auto is_match = def(false);
                        auto is_ready = def(false);
                        $if (is_in_range) {
                            auto frame_index = indices.read(index_offset + x);
                            auto token = coro_frame_read_field<uint>(
                                frame_buf, frame_index, frame_capacity,
                                layout, soa, 6u);
                            is_match = (token == current_queue);
                            is_ready = (ready_flags.read(x) != 0u);
                            $if (is_match) {
                                auto target_queue = select(
                                    retry_queue, next_queue, is_ready);
                                $if (target_queue != current_queue) {
                                    coro_frame_write_field(
                                        frame_buf, frame_index, frame_capacity,
                                        layout, soa, 6u, target_queue);
                                };
                            } $else {
                                if (fuse_continuation_counts) {
                                    $if ((current_queue != retry_queue) &
                                         (token != 0u) &
                                         (token < total_token_count)) {
                                        auto resume_count =
                                            Expr<Buffer<uint>>{*resume_count_buffer};
                                        resume_count.atomic(token).fetch_add(1u);
                                    };
                                }
                            };
                        };
                        if (fuse_continuation_counts) {
                            auto resume_count =
                                Expr<Buffer<uint>>{*resume_count_buffer};
                            $if ((x == 0u) & (current_queue != retry_queue)) {
                                resume_count.atomic(current_queue)
                                    .fetch_sub(count);
                            };
                            auto go_next = is_match & is_ready;
                            auto go_retry = is_match & (!is_ready);
                            if (warp_aggregate_counts) {
                                auto wn = warp_active_count_bits(go_next);
                                auto wr = warp_active_count_bits(go_retry);
                                $if (warp_lane_id() == 0u) {
                                    $if (wn != 0u) {
                                        block_counts.atomic(0u).fetch_add(wn);
                                    };
                                    $if (wr != 0u) {
                                        block_counts.atomic(1u).fetch_add(wr);
                                    };
                                };
                            } else {
                                $if (go_next) {
                                    block_counts.atomic(0u).fetch_add(1u);
                                } $else {
                                    $if (go_retry) {
                                        block_counts.atomic(1u).fetch_add(1u);
                                    };
                                };
                            }
                            sync_block();
                            $if (thread_x() == 0u) {
                                auto total_next = block_counts[0u];
                                auto total_retry = block_counts[1u];
                                $if (current_queue == retry_queue) {
                                    $if (total_next != 0u) {
                                        resume_count.atomic(current_queue)
                                            .fetch_sub(total_next);
                                        $if ((next_queue != 0u) &
                                             (next_queue < total_token_count)) {
                                            resume_count.atomic(next_queue)
                                                .fetch_add(total_next);
                                        };
                                    };
                                } $else {
                                    $if ((total_next != 0u) &
                                         (next_queue != 0u) &
                                         (next_queue < total_token_count)) {
                                        resume_count.atomic(next_queue)
                                            .fetch_add(total_next);
                                    };
                                    $if ((total_retry != 0u) &
                                         (retry_queue != 0u) &
                                         (retry_queue < total_token_count)) {
                                        resume_count.atomic(retry_queue)
                                            .fetch_add(total_retry);
                                    };
                                };
                            };
                        }
                    };
                _advance_extension_stage_filtered_shader = _compile_shader(
                    *_device, advance_extension_stage_filtered,
                    detail::coro_scheduler_shader_option(
                        _config.shader_option,
                        "wavefront_advance_extension_stage_filtered"),
                    "wavefront_advance_extension_stage_filtered");
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
        if (!_timeline_labels_registered && _device != nullptr &&
            _device->backend_name() == "vk" &&
            _device->impl()->query("vk_timeline_active") == "1") {
            auto label_buf = [&](uint64_t handle, const char *name) noexcept {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "vk_timeline_label:buffer:%llu:%s",
                              static_cast<unsigned long long>(handle), name);
                static_cast<void>(_device->impl()->query(buf));
            };
            label_buf(_resume_count.handle(), "wf_resume_count");
            label_buf(_resume_offset.handle(), "wf_resume_offset");
            label_buf(_global_buffer.handle(), "wf_global_buffer");
            _timeline_labels_registered = true;
        }
        auto parse_bool_env = [](const char *name, bool default_val) noexcept {
            if (const auto *raw = std::getenv(name)) {
                if (raw[0] == '0' && raw[1] == '\0') { return false; }
                if (raw[0] == '1' && raw[1] == '\0') { return true; }
            }
            return default_val;
        };
        bool has_blocking_async = false;
        for (const auto &reg : _extension_stages) {
            if (reg.is_blocking_async) {
                has_blocking_async = true;
                break;
            }
        }
        const bool online_timing_active = parse_bool_env(
            "LUISA_CORO_ONLINE_TIMING",
            _config.enable_online_timing && has_blocking_async);
        const bool time_budget_active =
            online_timing_active && has_blocking_async &&
            parse_bool_env("LUISA_CORO_TIME_BUDGET_SCHED",
                           _config.enable_time_budget_scheduling);
        const size_t total_timing_models =
            static_cast<size_t>(_queue_count) + _last_dispatch_stats.auxiliary_work.size();
        if (_timing_models.size() < total_timing_models) {
            _timing_models.resize(total_timing_models);
        }
        auto &timing_models = _timing_models;
        auto &t_sync_ms = _t_sync_ms;
        auto &sync_bubble_samples = _sync_bubble_samples;
        bool online_hw_timestamps = false;
        bool hw_timing_armed = false;
        uint64_t hw_ts_cbs = 0ull;
        luisa::string timing_enable_query;
        luisa::string timing_disable_query;
        luisa::string timing_read_query;
        uint64_t last_hw_cb_seq = 0ull;
        auto iteration_count = uint64_t{0u};
        size_t pending_timed_slot = total_timing_models;
        uint pending_timed_count = 0u;
        uint pending_timed_dispatches = 0u;
        bool pending_timed_was_overlap = false;
        bool just_waited_async = false;
        uint64_t time_budget_overlap_dispatches = 0u;
        uint64_t time_budget_overlap_waits = 0u;
        auto set_hw_timing_armed = [&](bool arm) noexcept {
            if (!online_hw_timestamps || _device == nullptr ||
                _device->impl() == nullptr || hw_timing_armed == arm) {
                return;
            }
            static_cast<void>(_device->impl()->query(
                arm ? timing_enable_query : timing_disable_query));
            hw_timing_armed = arm;
        };
        auto read_hw_cb_timing = [&](uint64_t &gpu_ns, uint64_t &gap_ns,
                                     uint64_t &seq) noexcept -> bool {
            if (!online_hw_timestamps || _device == nullptr) { return false; }
            const auto res = _device->impl()->query(timing_read_query);
            if (res.empty()) { return false; }
            char *p1 = nullptr;
            char *p2 = nullptr;
            char *p3 = nullptr;
            const uint64_t g = std::strtoull(res.c_str(), &p1, 10);
            if (p1 == res.c_str() || *p1 != ' ') { return false; }
            const uint64_t gap = std::strtoull(p1 + 1, &p2, 10);
            if (p2 == p1 + 1 || *p2 != ' ') { return false; }
            const uint64_t s = std::strtoull(p2 + 1, &p3, 10);
            if (p3 == p2 + 1) { return false; }
            gpu_ns = g;
            gap_ns = gap;
            seq = s;
            return true;
        };
        auto stamp_pending_timing = [&](size_t slot_idx, uint count,
                                        bool is_overlap) noexcept {
            if (!online_timing_active) { return; }
            pending_timed_slot = slot_idx;
            pending_timed_count = count;
            ++pending_timed_dispatches;
            pending_timed_was_overlap = is_overlap;
        };

        auto pending_commands = CommandList::create(16u);
        auto flush_pending_raw = [&]() noexcept {
            if (!pending_commands.empty()) {
                stream << pending_commands.commit();
            }
        };
        auto flush_pending = [&]() noexcept {
            if (!pending_commands.empty()) {
                set_hw_timing_armed(false);
                flush_pending_raw();
                pending_timed_slot = total_timing_models;
                pending_timed_count = 0u;
                pending_timed_dispatches = 0u;
                pending_timed_was_overlap = false;
            }
        };
        auto sync_pending = [&]() noexcept {
            const bool had_pending_commands = !pending_commands.empty();
            const bool need_model_sample =
                pending_timed_dispatches == 1u &&
                pending_timed_slot < timing_models.size() &&
                pending_timed_count != 0u &&
                (timing_models[pending_timed_slot].samples < 8u ||
                 (iteration_count & 63u) == 0u);
            const bool need_bubble_sample =
                !just_waited_async && sync_bubble_samples < 16u;
            const bool want_hw_ts =
                online_hw_timestamps && had_pending_commands &&
                (need_model_sample || need_bubble_sample);
            const bool measure_wall =
                had_pending_commands &&
                online_timing_active && !online_hw_timestamps &&
                !just_waited_async && pending_timed_dispatches == 1u &&
                pending_timed_slot < timing_models.size() &&
                pending_timed_count != 0u;
            const auto wall_t0 = measure_wall
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
            if (online_hw_timestamps && had_pending_commands) {
                set_hw_timing_armed(want_hw_ts);
                if (want_hw_ts) {
                    ++hw_ts_cbs;
                }
            }
            flush_pending_raw();
            stream.synchronize();
            if (online_timing_active) {
                if (online_hw_timestamps) {
                    if (want_hw_ts) {
                        uint64_t gpu_ns = 0ull;
                        uint64_t gap_ns = 0ull;
                        uint64_t seq = 0ull;
                        if (read_hw_cb_timing(gpu_ns, gap_ns, seq)) {
                            if (seq > last_hw_cb_seq &&
                                pending_timed_dispatches == 1u &&
                                pending_timed_slot < timing_models.size() &&
                                pending_timed_count != 0u) {
                                timing_models[pending_timed_slot].observe(
                                    pending_timed_count,
                                    static_cast<double>(gpu_ns) * 1e-6,
                                    pending_timed_was_overlap);
                            }
                            if (seq != last_hw_cb_seq) {
                                if (need_bubble_sample &&
                                    seq == last_hw_cb_seq + 1ull) {
                                    const double gap_ms =
                                        static_cast<double>(gap_ns) * 1e-6;
                                    if (gap_ms >= 0.01 && gap_ms <= 0.50) {
                                        t_sync_ms = std::clamp(
                                            0.90 * t_sync_ms + 0.10 * gap_ms,
                                            0.02, 0.40);
                                        ++sync_bubble_samples;
                                    }
                                }
                                last_hw_cb_seq = seq;
                            }
                        }
                    }
                    if (sync_bubble_samples >= 16u) {
                        set_hw_timing_armed(false);
                    }
                } else if (measure_wall) {
                    const double wall_ms =
                        static_cast<double>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - wall_t0)
                                .count()) *
                        1e-6;
                    const double adj_ms =
                        std::max(0.005, wall_ms - 0.5 * t_sync_ms);
                    timing_models[pending_timed_slot].observe(
                        pending_timed_count, adj_ms, pending_timed_was_overlap);
                }
                pending_timed_slot = total_timing_models;
                pending_timed_count = 0u;
                pending_timed_dispatches = 0u;
                pending_timed_was_overlap = false;
                just_waited_async = false;
            }
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
        _last_dispatch_stats.online_timing_active = online_timing_active;
        _last_dispatch_stats.online_timing_hw_timestamps = false;
        _last_dispatch_stats.online_sync_bubble_ms = t_sync_ms;
        _last_dispatch_stats.time_budget_overlap_dispatches = 0u;
        _last_dispatch_stats.time_budget_overlap_waits = 0u;
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
        struct OnlineTimingScopeGuard {
            DeviceInterface *impl{nullptr};
            luisa::string disable_cmd;
            ~OnlineTimingScopeGuard() noexcept {
                if (impl != nullptr && !disable_cmd.empty()) {
                    static_cast<void>(impl->query(disable_cmd));
                }
            }
        } online_timing_guard;
        if (online_timing_active && _device != nullptr &&
            _device->impl() != nullptr &&
            _device->backend_name() == "vk") {
            timing_enable_query =
                luisa::format("vk_stream_timing_enable:{}", stream.handle());
            timing_disable_query =
                luisa::format("vk_stream_timing_disable:{}", stream.handle());
            if (_device->impl()->query(timing_enable_query) == "1") {
                online_hw_timestamps = true;
                hw_timing_armed = true;
                online_timing_guard.impl = _device->impl();
                online_timing_guard.disable_cmd = timing_disable_query;
                timing_read_query =
                    luisa::format("vk_stream_timing_read:{}", stream.handle());
                uint64_t g0 = 0ull;
                uint64_t gap0 = 0ull;
                uint64_t s0 = 0ull;
                if (read_hw_cb_timing(g0, gap0, s0)) {
                    last_hw_cb_seq = s0;
                }
            }
        }
        if (report_stats) {
            _last_dispatch_stats.online_timing_hw_timestamps = online_hw_timestamps;
        }
        const auto skip_initialize =
            _config.frame_buffer_compaction &&
            _config.incremental_continuation_counts &&
            _active_frame_capacity <= _initialized_frame_capacity;
        if (!skip_initialize) {
            pending_commands << _initialize_shader(
                                    _config.thread_count,
                                    _active_frame_capacity,
                                    _resume_index)
                                    .dispatch(_active_frame_capacity);
            if (_config.frame_buffer_compaction &&
                _config.incremental_continuation_counts) {
                _initialized_frame_capacity =
                    std::max(_initialized_frame_capacity, _active_frame_capacity);
            }
        }
        _used_frame_count = 0u;

        auto nc = _resume_kernels.size();
        auto nq = static_cast<size_t>(_queue_count);
        auto ntok = static_cast<size_t>(_total_token_count);
        for (size_t i = 0u; i < ntok; ++i) {
            _host_count[i] =
                _config.incremental_continuation_counts ?
                    0u :
                    (i == 0u ? _active_frame_capacity : 0u);
            _host_offset[i] = i == 0u ? 0u : _active_frame_capacity;
        }
        if (_config.incremental_continuation_counts) {
            pending_commands << _clear_count_shader(
                                    _resume_count, static_cast<uint>(ntok))
                                    .dispatch(static_cast<uint>(ntok));
        } else {
            pending_commands << _resume_count.copy_from(
                luisa::span{_host_count.data(), _host_count.size()});
        }
        if (_async_slot_count != 0u) {
            pending_commands << _clear_count_shader(
                                    _async_slot_source_queue, _async_slot_count)
                                    .dispatch(_async_slot_count);
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
        const bool batch_queues_enabled = parse_bool_env(
            "LUISA_CORO_WAVEFRONT_BATCH_QUEUES", true);
        const bool post_batch_chain_enabled = parse_bool_env(
            "LUISA_CORO_WAVEFRONT_POST_BATCH_CHAIN", true);
        auto parse_uint_env = [](const char *name, uint default_val) noexcept -> uint {
            const char *v = std::getenv(name);
            if (v == nullptr || *v == '\0') { return default_val; }
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end == v || *end != '\0') { return default_val; }
            return static_cast<uint>(parsed);
        };
        const uint batch_min_count_override = parse_uint_env(
            "LUISA_CORO_WAVEFRONT_BATCH_MIN_COUNT", 0u);
        bool tail_unroll_threshold_set = false;
        uint tail_unroll_threshold_env = 0u;
        if (const char *v = std::getenv("LUISA_CORO_WAVEFRONT_TAIL_UNROLL_THRESHOLD");
            v != nullptr && *v != '\0') {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end != v && *end == '\0') {
                tail_unroll_threshold_set = true;
                tail_unroll_threshold_env = static_cast<uint>(parsed);
            }
        }
        const uint tail_unroll_threshold =
            tail_unroll_threshold_set ? tail_unroll_threshold_env : 2048u;
        uint effective_tail_threshold = tail_unroll_threshold;
        const uint tail_unroll_rounds_override = parse_uint_env(
            "LUISA_CORO_WAVEFRONT_TAIL_UNROLL_ROUNDS", 0u);
        auto compute_batch_min_count = [&](size_t cur_scan_count) noexcept -> uint {
            if (batch_min_count_override != 0u) {
                return batch_min_count_override;
            }
            return std::min<uint>(
                std::max<uint>(static_cast<uint>(cur_scan_count) / 64u, 4096u),
                16384u);
        };
        auto parse_pct_env = [](const char *name, uint default_pct) noexcept {
            if (const auto *raw = std::getenv(name)) {
                char *end = nullptr;
                const auto parsed = std::strtoul(raw, &end, 10);
                if (end != raw && *end == '\0' && parsed <= 100ul) {
                    return static_cast<uint>(parsed);
                }
            }
            return default_pct;
        };
        const uint proactive_async_pct =
            parse_pct_env("LUISA_CORO_ASYNC_PROACTIVE_PCT", 25u);
        const uint min_async_overlap_pct =
            parse_pct_env("LUISA_CORO_ASYNC_MIN_OVERLAP_PCT", 12u);
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
        auto gather_scan_count = uint64_t{0u};
        auto compact_scan_count = uint64_t{0u};
        auto generated_count = uint64_t{0u};
        auto resumed_count = uint64_t{0u};
        auto max_scan_count = 0u;
        auto max_active_count = 0u;
        luisa::vector<uint> logical_count(nq, 0u);
        luisa::vector<uint> logical_members(nq, 0u);
        luisa::vector<uint> deferred_target_count(nc, 0u);
        struct AsyncBatchSlot {
            bool active{false};
            uint64_t generation{0u};
            uint64_t submit_seq{0u};
            uint64_t submit_time_ns{0u};
            size_t stage_index{0u};
            uint parked_token{0u};
            uint source_queue{0u};
            uint next_queue{0u};
            uint frame_count{0u};
            uint deferred_target{0u};
            bool was_deferred{false};
            bool blocks_compaction{false};
            size_t async_domain_index{0u};
        };
        luisa::vector<AsyncBatchSlot> async_slots(_async_slot_count);
        luisa::vector<uint> domain_async_in_flight(_async_domain_names.size(), 0u);
        if (_domain_async_est_ms.size() < _async_domain_names.size()) {
            _domain_async_est_ms.resize(_async_domain_names.size(), 0.0);
        }
        if (_domain_async_samples.size() < _async_domain_names.size()) {
            _domain_async_samples.resize(_async_domain_names.size(), 0u);
        }
        auto &domain_async_est_ms = _domain_async_est_ms;
        auto &domain_async_samples = _domain_async_samples;
        auto steady_now_ns = []() noexcept -> uint64_t {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        };
        size_t next_slot_cursor = 0u;
        auto &next_async_generation = _next_async_generation;
        auto &next_async_submit_seq = _next_async_submit_seq;
        auto &continuation_dispatch_seq = _continuation_dispatch_seq;
        auto &async_consumer_dispatch_seq = _async_consumer_dispatch_seq;
        auto &auxiliary_dispatch_seq = _auxiliary_dispatch_seq;
        size_t in_flight_async_count = 0u;
        size_t in_flight_compaction_blocking_count = 0u;
        luisa::vector<uint> filtered_zero_streak(_extension_stages.size(), 0u);
        size_t pending_filtered_stage = _extension_stages.size();
        uint pending_filtered_next_queue = 0u;
        uint pending_filtered_prev_next_count = 0u;
        auto is_queue_async_unavailable = [&](size_t q) noexcept {
            if (q < nc) { return false; }
            const auto si = q - nc;
            const auto &reg = _extension_stages[si];
            if (!reg.is_blocking_async) { return false; }
            if (_async_slot_count == 0u || in_flight_async_count >= _async_slot_count) {
                return true;
            }
            if (q < _queue_deferred.size() && _queue_deferred[q] != 0u &&
                q < _queue_deferred_target.size()) {
                const auto target = _queue_deferred_target[q];
                if (target < _target_has_deferred_self_loop.size() &&
                    _target_has_deferred_self_loop[target] != 0u) {
                    for (const auto &slot : async_slots) {
                        if (slot.active && slot.was_deferred && slot.deferred_target == target) {
                            return true;
                        }
                    }
                }
            }
            if (reg.async_domain_index < domain_async_in_flight.size() &&
                reg.async_domain_index < _domain_max_async_slots.size()) {
                uint domain_limit = _domain_max_async_slots[reg.async_domain_index];
                if (reg.handler != nullptr) {
                    const uint dynamic_limit =
                        std::max<uint>(1u, reg.handler->current_async_in_flight_limit());
                    domain_limit = std::min(domain_limit, dynamic_limit);
                }
                if (domain_async_in_flight[reg.async_domain_index] >= domain_limit) {
                    return true;
                }
            }
            return false;
        };
        auto last_deferred_target = nc;
        auto queue_owner = [&](size_t queue) noexcept {
            // Independent Extensions have distinct scheduling identities;
            // only before-resume suffixes belong to their continuation.
            return _queue_continuation[queue] < nc ?
                       _queue_continuation[queue] : static_cast<uint>(queue);
        };
        auto recompute_host_offsets = [&]() noexcept {
            auto active_offset = 0u;
            for (size_t i = 0u; i < ntok; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
            }
        };
        const bool init_clean_counts =
            _config.incremental_continuation_counts && !verify_queues;
        bool main_counts_dirty = !init_clean_counts;
        luisa::vector<uint8_t> aux_counts_dirty(
            _auxiliary_work.size(), init_clean_counts ? uint8_t{0u} : uint8_t{1u});
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
            ++continuation_dispatch_seq;
            if ((continuation < _queue_is_async_consumer.size() &&
                 _queue_is_async_consumer[continuation] != 0u) ||
                (source_queue < _queue_is_async_consumer.size() &&
                 _queue_is_async_consumer[source_queue] != 0u)) {
                ++async_consumer_dispatch_seq;
            }
            resumed_count += count;
            main_counts_dirty = true;
            for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                if (continuation < _auxiliary_work[a].max_emitted_per_continuation.size() &&
                    _auxiliary_work[a].max_emitted_per_continuation[continuation] != 0u) {
                    aux_counts_dirty[a] = 1u;
                }
            }
            stamp_pending_timing(continuation, count, in_flight_async_count > 0u);
            if (report_stats) {
                auto &stat = _last_dispatch_stats.continuations[continuation];
                stat.dispatch_count++;
                stat.executed_count += count;
            }
        };
        auto can_produce_into = [&](size_t src_cont, size_t dst_cont) noexcept -> bool {
            if (src_cont >= _continuation_targets.size()) { return false; }
            const auto &targets = _continuation_targets[src_cont];
            return std::find(targets.begin(), targets.end(),
                             static_cast<uint>(dst_cont)) != targets.end();
        };
        auto queue_is_batchable_main = [&](size_t q) noexcept -> bool {
            if (q < 1u || q >= nq || _host_count[q] == 0u) { return false; }
            if (q < _queue_deferred.size() && _queue_deferred[q] != 0u) { return false; }
            if (is_queue_async_unavailable(q)) { return false; }
            const auto owner = queue_owner(q);
            if (owner >= nc || logical_members[owner] != 1u) { return false; }
            if (_has_hint_sort && owner < _have_hint.size() && _have_hint[owner]) {
                return false;
            }
            if (q >= nc) {
                const auto &reg = _extension_stages[q - nc];
                if (reg.is_blocking_async || reg.before_resume_chain.empty()) {
                    return false;
                }
                for (auto idx : reg.before_resume_chain) {
                    if (_extension_stages[idx].handler == nullptr ||
                        !_extension_stages[idx].handler->supports_command_list()) {
                        return false;
                    }
                }
            }
            return true;
        };
        bool tail_unroll_graph_eligible =
            _tail_unroll_supported &&
            tail_unroll_threshold != 0u &&
            _config.largest_continuation_first &&
            _config.incremental_continuation_counts &&
            _config.fused_continuation_counts &&
            !_has_hint_sort;
        if (tail_unroll_graph_eligible) {
            for (const auto &reg : _extension_stages) {
                if (reg.handler != nullptr &&
                    (reg.is_blocking_async ||
                     reg.handler->execution() != WavefrontCoroExtensionExecution::before_resume)) {
                    tail_unroll_graph_eligible = false;
                    break;
                }
            }
        }
        if (tail_unroll_graph_eligible) {
            for (const auto d : _queue_deferred) {
                if (d != 0u) {
                    tail_unroll_graph_eligible = false;
                    break;
                }
            }
        }
        luisa::vector<uint8_t> tail_queue_used(nq, 0u);
        luisa::vector<luisa::vector<size_t>> cont_dst_queues(nc);
        luisa::vector<size_t> tail_queues;
        luisa::vector<uint> tail_aux_bound_per_round(_auxiliary_work.size(), 0u);
        if (_tail_unroll_supported) {
            for (size_t s = 0u; s < nc; ++s) {
                if (s >= _continuation_targets.size()) { continue; }
                for (const uint target : _continuation_targets[s]) {
                    size_t dst_q = static_cast<size_t>(target);
                    if (s < _extension_routes.size()) {
                        for (const auto &route : _extension_routes[s]) {
                            bool matches_target = false;
                            for (const auto &boundary : _extension_boundaries) {
                                if (boundary.index == route.boundary_index &&
                                    boundary.to_index == target) {
                                    matches_target = true;
                                    break;
                                }
                            }
                            if (matches_target &&
                                route.boundary_index < _extension_route_table.size() &&
                                _extension_route_table[route.boundary_index] != 0u) {
                                dst_q = static_cast<size_t>(
                                    _extension_route_table[route.boundary_index]);
                                break;
                            }
                        }
                    }
                    if (dst_q >= 1u && dst_q < nq) {
                        auto &dsts = cont_dst_queues[s];
                        if (std::find(dsts.begin(), dsts.end(), dst_q) == dsts.end()) {
                            dsts.emplace_back(dst_q);
                        }
                    }
                }
            }
        }
        if (tail_unroll_graph_eligible) {
            for (size_t s = 0u; s < nc; ++s) {
                for (const size_t dst_q : cont_dst_queues[s]) {
                    tail_queue_used[dst_q] = 1u;
                }
            }
            for (size_t q = 1u; q < nq; ++q) {
                if (tail_queue_used[q] == 0u) { continue; }
                const size_t cont = _queue_continuation[q];
                if (cont == 0u || cont >= nc || !_resume_kernels[cont]) {
                    tail_unroll_graph_eligible = false;
                    break;
                }
                tail_queues.emplace_back(q);
            }
            if (tail_unroll_graph_eligible) {
                std::sort(tail_queues.begin(), tail_queues.end(),
                          [&](size_t lhs, size_t rhs) noexcept {
                              const auto lc = _queue_continuation[lhs];
                              const auto rc = _queue_continuation[rhs];
                              return lc != rc ? lc < rc : lhs < rhs;
                          });
                if (!tail_unroll_threshold_set) {
                    effective_tail_threshold =
                        tail_queues.size() <= 6u ? 16384u : 8192u;
                }
                for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                    const auto &reg = _auxiliary_work[a];
                    uint max_bound = 0u;
                    uint sum_bound = 0u;
                    bool has_forward_chain_to_aux = false;
                    for (size_t i = 0u; i < tail_queues.size(); ++i) {
                        const size_t ci = _queue_continuation[tail_queues[i]];
                        const uint bi = ci < reg.max_emitted_per_continuation.size() ?
                                            reg.max_emitted_per_continuation[ci] :
                                            0u;
                        if (bi == 0u) { continue; }
                        max_bound = std::max(max_bound, bi);
                        sum_bound += bi;
                        for (size_t j = i; j < tail_queues.size(); ++j) {
                            const size_t cj = _queue_continuation[tail_queues[j]];
                            const uint bj = cj < reg.max_emitted_per_continuation.size() ?
                                                reg.max_emitted_per_continuation[cj] :
                                                0u;
                            if (bj != 0u && can_produce_into(ci, cj)) {
                                has_forward_chain_to_aux = true;
                            }
                        }
                    }
                    tail_aux_bound_per_round[a] =
                        has_forward_chain_to_aux ? sum_bound : max_bound;
                }
            }
        }
        auto encode_single_member_main_queue = [&](size_t q, uint cur_scan_count,
                                                   const WavefrontCoroSchedulerSnapshot &snap) noexcept -> bool {
            const auto count = _host_count[q];
            const auto continuation = _queue_continuation[q];
            bool fused_extension_gather = false;
            bool use_identity_selected = false;
            if (q >= nc) {
                const auto &ext_stage = _extension_stages[q - nc];
                if (!ext_stage.before_resume_chain.empty()) {
                    const bool has_hint = _has_hint_sort &&
                                          continuation < _have_hint.size() &&
                                          _have_hint[continuation];
                    if (!has_hint) {
                        const auto &first_handler =
                            *_extension_stages[ext_stage.before_resume_chain.front()].handler;
                        fused_extension_gather =
                            first_handler.gathers_selected_frames_for(
                                count, cur_scan_count);
                    }
                }
            } else if (count == cur_scan_count &&
                       !(_has_hint_sort && q < _have_hint.size() && _have_hint[q])) {
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
                                          ((cur_scan_count + 1023u) / 1024u) * 256u :
                                          cur_scan_count;
                pending_commands << _gather_selected_shader(
                                        _config.thread_count, _resume_index,
                                        _global_buffer, count_slot,
                                        static_cast<uint>(q), cur_scan_count)
                                        .dispatch(gather_threads);
                if (verify_queues) {
                    uint gathered_count = 0u;
                    pending_commands << _global_buffer.view()
                                            .subview(count_slot, 1u)
                                            .copy_to(luisa::span{
                                                &gathered_count, 1u});
                    luisa::vector<uint> gathered_indices(count);
                    pending_commands << _resume_index.view()
                                            .subview(0u, count)
                                            .copy_to(luisa::span{gathered_indices});
                    sync_pending();
                    LUISA_ASSERT(
                        gathered_count == count,
                        "Incremental wavefront selected-queue gather "
                        "violation at iteration {}, continuation {}: "
                        "gathered {} frames, expected {}.",
                        iteration_count, q, gathered_count,
                        count);
                    luisa::vector<uint8_t> seen(_config.thread_count, 0u);
                    for (auto slot = 0u; slot < count; ++slot) {
                        auto idx = gathered_indices[slot];
                        LUISA_ASSERT(idx < _config.thread_count && seen[idx] == 0u,
                                     "Gather duplicate/invalid frame index {} at slot {} (iter {}).",
                                     idx, slot, iteration_count);
                        seen[idx] = 1u;
                    }
                }
                gather_scan_count += cur_scan_count;
            }
            _host_offset[q] = 0u;
            if (q >= nc) {
                const auto &registered = _extension_stages[q - nc];
                auto indices = _resume_index.view().subview(0u, count);
                bool ungathered_head = fused_extension_gather;
                for (auto index : registered.before_resume_chain) {
                    auto &stage = _extension_stages[index];
                    CommandList *handler_commands = &pending_commands;
                    const bool is_ungathered = ungathered_head;
                    ungathered_head = false;
                    {
                        detail::WavefrontCoroSchedulerSnapshotScope snapshot_scope{snap};
                        indices = stage.handler->dispatch_queue(WavefrontCoroExtensionDispatchContext{
                            .stream = stream,
                            .frame_buffer = _frame_buffer.view(),
                            .frame_indices = indices,
                            .frame_count = count,
                            .frame_capacity = _config.thread_count,
                            .logical_dispatch_size = dispatch_size,
                            .stage = stage.stage,
                            .command_list = handler_commands,
                            .source_queue_token = static_cast<uint>(q),
                            .scan_count = cur_scan_count,
                            .ungathered_scan = is_ungathered,
                            .scheduler_snapshot = snap});
                    }
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
                }
                resume(continuation, static_cast<uint>(q), indices, count);
            } else {
                auto indices = use_identity_selected ?
                                   _identity_index.view() :
                                   _resume_index.view();
                resume(continuation, static_cast<uint>(q), indices, count);
            }
            return continuation < deterministic_sink_continuation.size() &&
                   deterministic_sink_continuation[continuation];
        };
        auto encode_aux_stage = [&](size_t aux_idx, uint aux_stage,
                                    const WavefrontCoroSchedulerSnapshot &snap) noexcept -> bool {
            auto &&registered = _auxiliary_work[aux_idx];
            auto count = registered.work->stage_host_count(aux_stage);
            LUISA_ASSERT(count != 0u,
                         "Selected an empty wavefront auxiliary "
                         "queue '{}'.",
                         registered.work->name());
            {
                detail::WavefrontCoroSchedulerSnapshotScope snapshot_scope{snap};
                registered.work->encode_dispatch_stage(aux_stage, stream, pending_commands, args...);
            }
            ++auxiliary_dispatch_seq;
            aux_counts_dirty[aux_idx] = 1u;
            stamp_pending_timing(
                static_cast<size_t>(nq) + registered.stats_offset + aux_stage,
                count, in_flight_async_count > 0u);
            if (report_stats) {
                auto &work = _last_dispatch_stats
                                 .auxiliary_work[registered.stats_offset + aux_stage];
                work.dispatch_count++;
                work.executed_count += count;
            }
            if (_config.incremental_continuation_counts && !verify_queues &&
                registered.work->try_predict_stage_transition(aux_stage)) {
                aux_counts_dirty[aux_idx] = 0u;
                return true;
            }
            return false;
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
            bool any_aux_dirty = false;
            for (auto d : aux_counts_dirty) {
                if (d != 0u) {
                    any_aux_dirty = true;
                    break;
                }
            }
            auto skip_readback =
                _config.incremental_continuation_counts &&
                counts_known_on_host && !verify_queues && !any_aux_dirty;
            bool need_sync_readback = false;
            if (!skip_readback) {
                for (size_t i = 0u; i < _auxiliary_work.size(); ++i) {
                    if (verify_queues || !_config.incremental_continuation_counts ||
                        aux_counts_dirty[i] != 0u) {
                        _auxiliary_work[i].work->encode_count_readback(stream, pending_commands);
                        aux_counts_dirty[i] = 0u;
                        need_sync_readback = true;
                    }
                }
            }
            if (_config.incremental_continuation_counts) {
                if (!skip_readback) {
                    if (verify_queues || main_counts_dirty) {
                        pending_commands << _resume_count.copy_to(
                            luisa::span{_host_count.data(),
                                        _host_count.size()});
                        main_counts_dirty = false;
                        need_sync_readback = true;
                    }
                    if (need_sync_readback) {
                        sync_pending();
                    }
                }
                counts_known_on_host = false;
                if (verify_queues && scan_count != 0u) {
                    pending_commands << _clear_count_shader(
                                            _resume_offset, static_cast<uint>(ntok))
                                            .dispatch(static_cast<uint>(ntok));
                    pending_commands << _count_shader(
                                            _config.thread_count,
                                            _resume_offset, scan_count)
                                            .dispatch(scan_count);
                    luisa::vector<uint> actual(ntok);
                    pending_commands << _resume_offset.copy_to(luisa::span{actual});
                    sync_pending();
                    for (size_t i = 1u; i < ntok; ++i) {
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
                                        _resume_count, static_cast<uint>(ntok))
                                        .dispatch(static_cast<uint>(ntok));
                if (scan_count != 0u) {
                    pending_commands << _count_shader(
                                            _config.thread_count,
                                            _resume_count, scan_count)
                                            .dispatch(scan_count);
                }
                pending_commands << _resume_count.copy_to(luisa::span{_host_count.data(), _host_count.size()});
                sync_pending();
            }

            if (pending_filtered_stage < _extension_stages.size()) {
                const uint new_next =
                    pending_filtered_next_queue < nq ?
                        _host_count[pending_filtered_next_queue] :
                        0u;
                if (new_next <= pending_filtered_prev_next_count) {
                    ++filtered_zero_streak[pending_filtered_stage];
                } else {
                    filtered_zero_streak[pending_filtered_stage] = 0u;
                }
                pending_filtered_stage = _extension_stages.size();
            }

            auto active_count = 0u;
            std::fill(logical_count.begin(), logical_count.end(), 0u);
            std::fill(logical_members.begin(), logical_members.end(), 0u);
            for (size_t i = 1u; i < ntok; ++i) {
                active_count += _host_count[i];
                if (i < nq) {
                    auto owner = queue_owner(i);
                    logical_count[owner] += _host_count[i];
                    logical_members[owner] += static_cast<uint>(_host_count[i] != 0u);
                    if (report_stats && i >= nc) {
                        auto &extension = _last_dispatch_stats.extensions[i - nc];
                        extension.peak_queued_count = std::max(
                            extension.peak_queued_count, _host_count[i]);
                    }
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
            auto eligible_deferred_count = 0u;
            auto eligible_async_deferred_count = 0u;
            auto deferred_blocker_frames = 0u;
            auto deferred_blocker_nonempty = false;
            auto eligible_blocker_nonempty = false;
            for (size_t i = 1u; i < nq; ++i) {
                if (i < _queue_deferred.size() && _queue_deferred[i] != 0u) {
                    deferred_count += _host_count[i];
                    if (!is_queue_async_unavailable(i)) {
                        eligible_deferred_count += _host_count[i];
                        if (i >= nc && _extension_stages[i - nc].is_blocking_async) {
                            eligible_async_deferred_count += _host_count[i];
                        }
                    }
                } else if (_host_count[i] != 0u &&
                           i < _queue_blocks_deferred.size() &&
                           _queue_blocks_deferred[i] != 0u) {
                    deferred_blocker_frames += _host_count[i];
                    deferred_blocker_nonempty = true;
                    if (!is_queue_async_unavailable(i)) {
                        eligible_blocker_nonempty = true;
                    }
                }
            }
            for (const auto &slot : async_slots) {
                if (slot.active && slot.was_deferred && slot.parked_token < ntok) {
                    deferred_count += _host_count[slot.parked_token];
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
                    bool stage_is_async_blocked = false;
                    for (size_t d = 0u; d < domain_async_in_flight.size(); ++d) {
                        if (domain_async_in_flight[d] != 0u &&
                            registered.work->stage_blocks_async_domain(
                                stage, _async_domain_names[d])) {
                            stage_is_async_blocked = true;
                            break;
                        }
                    }
                    if (!stage_is_async_blocked &&
                        stage_count > selected_auxiliary_count) {
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

            recompute_host_offsets();
            if (dispatch_counter == N && active_count == 0u &&
                auxiliary_active_count == 0u && in_flight_async_count == 0u) {
                if (report_stats) {
                    sync_pending();
                } else {
                    flush_pending();
                }
                break;
            }
            bool buckets_materialized_this_iter = _config.gather_by_sorting;
            auto gather_queue_indices = [&](size_t q, uint count,
                                            uint cur_scan_count) noexcept -> BufferView<uint> {
                if (count == 0u) {
                    return _resume_index.view().subview(0u, 0u);
                }
                if (_config.incremental_continuation_counts) {
                    auto count_slot = global_count_slot;
                    global_count_slot ^= 1u;
                    auto gather_threads = _gather_selected_4x ?
                                              ((cur_scan_count + 1023u) / 1024u) * 256u :
                                              cur_scan_count;
                    pending_commands << _gather_selected_shader(
                                            _config.thread_count, _resume_index,
                                            _global_buffer, count_slot,
                                            static_cast<uint>(q), cur_scan_count)
                                            .dispatch(gather_threads);
                    gather_scan_count += cur_scan_count;
                    return _resume_index.view().subview(0u, count);
                }
                if (_config.gather_by_sorting) {
                    if (!buckets_materialized_this_iter && cur_scan_count != 0u) {
                        _encode_sort_token_buckets(pending_commands, stream, cur_scan_count);
                        buckets_materialized_this_iter = true;
                    }
                    return _resume_index.view().subview(_host_offset[q], count);
                }
                pending_commands << _resume_offset.copy_from(
                    luisa::span{_host_offset.data(), _host_offset.size()});
                if (cur_scan_count != 0u) {
                    pending_commands << _gather_shader(
                                            _config.thread_count,
                                            _resume_index, _resume_offset, cur_scan_count)
                                            .dispatch(cur_scan_count);
                }
                return _resume_index.view().subview(_host_offset[q], count);
            };
            auto advance_dispatch_threads = [](uint n) noexcept {
                return ((n + 255u) / 256u) * 256u;
            };
            auto try_complete_async_ticket = [&](const WavefrontCoroAsyncBatchTicket &ticket,
                                                 uint cur_scan_count,
                                                 const WavefrontCoroSchedulerSnapshot &snap,
                                                 bool *used_filtered_out = nullptr,
                                                 bool from_wait = false) noexcept -> bool {
                if (used_filtered_out != nullptr) { *used_filtered_out = false; }
                if (!ticket.valid() || ticket.slot_index >= async_slots.size()) {
                    return false;
                }
                auto &slot = async_slots[ticket.slot_index];
                if (!slot.active || slot.generation != ticket.generation) {
                    return false;
                }
                if (online_timing_active &&
                    slot.async_domain_index < domain_async_est_ms.size() &&
                    slot.submit_time_ns != 0ull) {
                    const uint64_t now_ns = steady_now_ns();
                    const double elapsed_ms =
                        now_ns >= slot.submit_time_ns
                            ? std::clamp(
                                  static_cast<double>(now_ns - slot.submit_time_ns) * 1e-6,
                                  0.02, 100.0)
                            : 0.02;
                    const size_t dom = slot.async_domain_index;
                    if (from_wait) {
                        domain_async_est_ms[dom] =
                            (domain_async_samples[dom] == 0u)
                                ? elapsed_ms
                                : (0.80 * domain_async_est_ms[dom] + 0.20 * elapsed_ms);
                        ++domain_async_samples[dom];
                    } else if (domain_async_samples[dom] == 0u) {
                        domain_async_est_ms[dom] = elapsed_ms;
                        ++domain_async_samples[dom];
                    } else if (elapsed_ms < domain_async_est_ms[dom]) {
                        domain_async_est_ms[dom] =
                            0.90 * domain_async_est_ms[dom] + 0.10 * elapsed_ms;
                        ++domain_async_samples[dom];
                    }
                }
                auto &reg = _extension_stages[slot.stage_index];
                const uint parked_tok = slot.parked_token;
                const uint count = _host_count[parked_tok];
                BufferView<uint> indices;
                if (reg.modifies_frame) {
                    indices = gather_queue_indices(parked_tok, count, cur_scan_count);
                }
                {
                    detail::WavefrontCoroSchedulerSnapshotScope snapshot_scope{snap};
                    reg.handler->publish_async(WavefrontCoroAsyncPublishContext{
                        .stream = stream,
                        .command_list = pending_commands,
                        .frame_buffer = _frame_buffer.view(),
                        .frame_indices = indices,
                        .frame_count = count,
                        .frame_capacity = _config.thread_count,
                        .stage = reg.stage,
                        .scheduler_snapshot = snap,
                        .async_ticket = ticket});
                }
                if (!reg.modifies_frame) {
                    indices = gather_queue_indices(parked_tok, count, cur_scan_count);
                }
                bool use_filtered = false;
                if (count != 0u &&
                    reg.handler != nullptr &&
                    reg.handler->supports_filtered_advance() &&
                    static_cast<bool>(_advance_extension_stage_filtered_shader)) {
                    if (filtered_zero_streak[slot.stage_index] >= 2u) {
                        filtered_zero_streak[slot.stage_index] = 0u;
                    } else {
                        use_filtered = true;
                    }
                }
                if (count != 0u) {
                    if (use_filtered) {
                        reg.handler->encode_advance_filter(WavefrontCoroAdvanceFilterContext{
                            .stream = stream,
                            .command_list = pending_commands,
                            .frame_buffer = _frame_buffer.view(),
                            .frame_indices = indices,
                            .ready_flags = _advance_ready_flags.view(0u, count),
                            .frame_count = count,
                            .frame_capacity = _config.thread_count,
                            .stage = reg.stage});
                        pending_commands << _advance_extension_stage_filtered_shader(
                                                _config.thread_count, indices,
                                                0u, count, parked_tok,
                                                slot.next_queue, slot.source_queue,
                                                _advance_ready_flags)
                                                .dispatch(advance_dispatch_threads(count));
                    } else {
                        pending_commands << _advance_extension_stage_shader(
                                                _config.thread_count, indices,
                                                0u, count, parked_tok,
                                                slot.next_queue)
                                                .dispatch(advance_dispatch_threads(count));
                    }
                    if (_config.incremental_continuation_counts &&
                        !_config.fused_continuation_counts) {
                        pending_commands << _publish_resumed_count_shader(
                                                _config.thread_count, indices, 0u,
                                                _resume_count,
                                                parked_tok, count)
                                                .dispatch(count);
                    }
                }
                const size_t q = slot.source_queue;
                if (q < _queue_deferred.size() && _queue_deferred[q] != 0u &&
                    q < _queue_deferred_target.size()) {
                    last_deferred_target = _queue_deferred_target[q];
                }
                if (report_stats) {
                    auto &stat = _last_dispatch_stats.extensions[slot.stage_index];
                    stat.dispatch_count++;
                    stat.executed_count += count;
                    stat.peak_queued_count = std::max(stat.peak_queued_count, count);
                }
                _last_dispatch_stats.extension_count += count;
                if (slot.async_domain_index < domain_async_in_flight.size() &&
                    domain_async_in_flight[slot.async_domain_index] > 0u) {
                    --domain_async_in_flight[slot.async_domain_index];
                }
                if (slot.blocks_compaction && in_flight_compaction_blocking_count > 0u) {
                    --in_flight_compaction_blocking_count;
                }
                if (in_flight_async_count > 0u) {
                    --in_flight_async_count;
                }
                _host_count[parked_tok] = 0u;
                if (use_filtered) {
                    pending_filtered_stage = slot.stage_index;
                    pending_filtered_next_queue = slot.next_queue;
                    pending_filtered_prev_next_count =
                        slot.next_queue < nq ? _host_count[slot.next_queue] : 0u;
                    if (_config.incremental_continuation_counts) {
                        recompute_host_offsets();
                    }
                    slot = AsyncBatchSlot{};
                    counts_known_on_host = false;
                    main_counts_dirty = true;
                    if (used_filtered_out != nullptr) {
                        *used_filtered_out = true;
                    }
                } else {
                    if (slot.next_queue != 0u && slot.next_queue < nq) {
                        _host_count[slot.next_queue] += count;
                    }
                    if (_config.incremental_continuation_counts) {
                        recompute_host_offsets();
                    }
                    slot = AsyncBatchSlot{};
                    counts_known_on_host =
                        (_config.incremental_continuation_counts &&
                         _config.fused_continuation_counts && !verify_queues);
                    if (!counts_known_on_host) {
                        main_counts_dirty = true;
                    }
                }
                if (online_timing_active) {
                    pending_timed_dispatches += 2u;
                    pending_timed_slot = total_timing_models;
                }
                return true;
            };
            auto notify_active_async_handlers = [&](const WavefrontCoroSchedulerSnapshot &snap,
                                                    bool waiting_for_progress) noexcept {
                if (in_flight_async_count == 0u) { return; }
                luisa::vector<ExtensionHandler *> notified;
                notified.reserve(async_slots.size());
                for (const auto &slot : async_slots) {
                    if (!slot.active) { continue; }
                    auto *handler = _extension_stages[slot.stage_index].handler.get();
                    if (handler == nullptr) { continue; }
                    if (std::find(notified.begin(), notified.end(), handler) != notified.end()) {
                        continue;
                    }
                    notified.push_back(handler);
                    handler->on_async_scheduler_progress(snap, waiting_for_progress);
                }
            };
            auto poll_all_ready_async = [&](uint cur_scan_count,
                                            const WavefrontCoroSchedulerSnapshot &snap) noexcept -> bool {
                if (in_flight_async_count == 0u) { return false; }
                bool any_published = false;
                const size_t max_rounds = _async_slot_count * 4u + 4u;
                for (size_t round = 0u; round < max_rounds && in_flight_async_count > 0u; ++round) {
                    bool round_progressed = false;
                    luisa::vector<ExtensionHandler *> rejected_handlers;
                    for (size_t s = 0u; s < async_slots.size(); ++s) {
                        if (!async_slots[s].active) { continue; }
                        auto *handler = _extension_stages[async_slots[s].stage_index].handler.get();
                        if (handler == nullptr) { continue; }
                        if (std::find(rejected_handlers.begin(), rejected_handlers.end(), handler) !=
                            rejected_handlers.end()) {
                            continue;
                        }
                        WavefrontCoroAsyncBatchTicket ticket{};
                        if (!handler->poll_async_ready(ticket)) {
                            continue;
                        }
                        if (!ticket.valid()) {
                            ticket = WavefrontCoroAsyncBatchTicket{
                                .slot_index = static_cast<uint>(s),
                                .generation = async_slots[s].generation};
                        }
                        bool used_filtered = false;
                        if (try_complete_async_ticket(
                                ticket, cur_scan_count, snap, &used_filtered)) {
                            any_published = true;
                            round_progressed = true;
                            if (used_filtered ||
                                !_config.incremental_continuation_counts) {
                                return true;
                            }
                            break;
                        } else {
                            rejected_handlers.push_back(handler);
                        }
                    }
                    if (!round_progressed) {
                        break;
                    }
                }
                return any_published;
            };
            auto wait_and_complete_one_async = [&](uint cur_scan_count,
                                                   const WavefrontCoroSchedulerSnapshot &snap,
                                                   size_t domain_filter) noexcept -> bool {
                flush_pending();
                notify_active_async_handlers(snap, true);
                if (poll_all_ready_async(cur_scan_count, snap)) {
                    return true;
                }
                size_t target_slot = async_slots.size();
                uint64_t best_seq = std::numeric_limits<uint64_t>::max();
                for (size_t s = 0u; s < async_slots.size(); ++s) {
                    const auto &slot = async_slots[s];
                    if (!slot.active) { continue; }
                    if (domain_filter < _async_domain_names.size() &&
                        slot.async_domain_index != domain_filter) {
                        continue;
                    }
                    if (slot.submit_seq < best_seq) {
                        best_seq = slot.submit_seq;
                        target_slot = s;
                    }
                }
                if (target_slot >= async_slots.size()) {
                    return false;
                }
                auto *handler = _extension_stages[async_slots[target_slot].stage_index].handler.get();
                WavefrontCoroAsyncBatchTicket ticket{};
                just_waited_async = true;
                if (!handler->wait_async_any(ticket)) {
                    return poll_all_ready_async(cur_scan_count, snap);
                }
                if (!ticket.valid()) {
                    ticket = WavefrontCoroAsyncBatchTicket{
                        .slot_index = static_cast<uint>(target_slot),
                        .generation = async_slots[target_slot].generation};
                }
                bool used_filtered = false;
                bool completed = try_complete_async_ticket(
                    ticket, cur_scan_count, snap, &used_filtered, true);
                if (!used_filtered && _config.incremental_continuation_counts) {
                    completed |= poll_all_ready_async(cur_scan_count, snap);
                }
                return completed;
            };
            if (in_flight_async_count > 0u) {
                const auto early_refill_threshold =
                    _config.refill_threshold == 0u ?
                        _active_frame_capacity / 2u :
                        std::min(_config.refill_threshold, _active_frame_capacity);
                const WavefrontCoroSchedulerSnapshot early_snapshot{
                    .iteration = iteration_count,
                    .active_frames = active_count,
                    .deferred_frames = deferred_count,
                    .non_deferred_frames =
                        active_count >= deferred_count ? active_count - deferred_count : 0u,
                    .non_deferred_largest = 0u,
                    .deferred_blocker_frames = deferred_blocker_frames,
                    .auxiliary_ready_frames = auxiliary_active_count,
                    .unspawned_threads =
                        N >= dispatch_counter ? N - dispatch_counter : 0u,
                    .empty_frame_slots = empty_count,
                    .active_frame_capacity = _active_frame_capacity,
                    .refill_threshold = early_refill_threshold,
                    .deferred_backpressure =
                        deferred_count >= std::max(early_refill_threshold / 2u, 1u),
                    .valid = true,
                    .continuation_dispatch_seq = continuation_dispatch_seq,
                    .async_consumer_dispatch_seq = async_consumer_dispatch_seq,
                    .auxiliary_dispatch_seq = auxiliary_dispatch_seq,
                    .in_flight_async_batches = static_cast<uint>(in_flight_async_count),
                    .max_async_batch_slots = _async_slot_count};
                notify_active_async_handlers(early_snapshot, false);
                if (poll_all_ready_async(scan_count, early_snapshot)) {
                    continue;
                }
            }

            auto selected = nq;
            auto selected_count = 0u;
            auto selected_priority = nq;
            auto select_queue = [&](bool allow_deferred) noexcept {
                selected = nq;
                selected_count = 0u;
                selected_priority = nq;
                const auto select_refill_threshold =
                    _config.refill_threshold == 0u ?
                        _active_frame_capacity / 2u :
                        std::min(_config.refill_threshold, _active_frame_capacity);
                const bool refill_wanted =
                    !allow_deferred && dispatch_counter < N &&
                    active_count < select_refill_threshold;
                size_t best_nc_priority = nq;
                uint best_nc_count = 0u;
                size_t best_c_priority = nq;
                uint best_c_count = 0u;
                for (size_t i = 1u; i < nq; ++i) {
                    auto is_deferred =
                        i < _queue_deferred.size() && _queue_deferred[i] != 0u;
                    if (is_deferred != allow_deferred) { continue; }
                    if (is_queue_async_unavailable(i)) {
                        continue;
                    }
                    const auto c = logical_count[i];
                    if (c == 0u) { continue; }
                    const bool is_collect =
                        i < _collect_at.size() && _collect_at[i];
                    if (is_collect) {
                        if (c > best_c_count) {
                            best_c_count = c;
                            best_c_priority = i;
                        }
                    } else {
                        if (c > best_nc_count) {
                            best_nc_count = c;
                            best_nc_priority = i;
                        }
                    }
                }
                if (!allow_deferred && best_nc_count > 0u) {
                    const uint select_batch_min =
                        (batch_queues_enabled &&
                         _config.largest_continuation_first &&
                         _config.incremental_continuation_counts &&
                         _config.fused_continuation_counts &&
                         !online_timing_active &&
                         scan_count != 0u) ?
                            compute_batch_min_count(scan_count) :
                            1u;
                    const bool defer_tiny_nc_for_collect =
                        best_c_count >= select_batch_min &&
                        best_nc_count < select_batch_min &&
                        static_cast<uint64_t>(best_c_count) >=
                            static_cast<uint64_t>(best_nc_count) * 4ull;
                    if (best_c_count > 0u &&
                        (defer_tiny_nc_for_collect ||
                         (refill_wanted &&
                          best_c_priority < _refill_at.size() &&
                          _refill_at[best_c_priority] &&
                          best_c_count >= best_nc_count))) {
                        selected_count = best_c_count;
                        selected_priority = best_c_priority;
                    } else {
                        selected_count = best_nc_count;
                        selected_priority = best_nc_priority;
                    }
                } else if (best_c_count > best_nc_count) {
                    selected_count = best_c_count;
                    selected_priority = best_c_priority;
                } else {
                    selected_count = best_nc_count;
                    selected_priority = best_nc_priority;
                }
                if (selected_count != 0u) {
                    for (size_t i = 1u; i < nq; ++i) {
                        if (_host_count[i] != 0u &&
                            !is_queue_async_unavailable(i) &&
                            queue_owner(i) == selected_priority) {
                            selected = i;
                            break;
                        }
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
                    if (is_queue_async_unavailable(i)) {
                        continue;
                    }
                    if (logical_count[i] > selected_count) {
                        selected_count = logical_count[i];
                        selected_priority = i;
                    }
                }
                for (size_t i = 1u; i < nq; ++i) {
                    if (_host_count[i] != 0u &&
                        !is_queue_async_unavailable(i) &&
                        queue_owner(i) == selected_priority) {
                        selected = i;
                        break;
                    }
                }
            };
            auto select_deferred_queue = [&](bool async_only = false) noexcept {
                selected = nq;
                selected_count = 0u;
                selected_priority = nq;
                std::fill(deferred_target_count.begin(), deferred_target_count.end(), 0u);
                for (size_t i = nc; i < nq; ++i) {
                    if (i < _queue_deferred.size() && _queue_deferred[i] != 0u &&
                        !is_queue_async_unavailable(i) &&
                        (!async_only || _extension_stages[i - nc].is_blocking_async)) {
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
                        _host_count[i] == 0u ||
                        is_queue_async_unavailable(i) ||
                        (async_only && !_extension_stages[i - nc].is_blocking_async)) {
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
                const bool handler_prefers_entry =
                    best_entry != nq &&
                    _extension_stages[best_entry - nc].handler != nullptr &&
                    _extension_stages[best_entry - nc].handler->prefer_deferred_entry_drain();
                selected = (best_entry != nq &&
                            (best_self == nq ||
                             handler_prefers_entry ||
                             static_cast<uint64_t>(best_entry_count) * 8ull >=
                                 static_cast<uint64_t>(best_self_count))) ?
                               best_entry :
                               best_self;
                if (selected < nq) {
                    selected_count = _host_count[selected];
                    selected_priority = selected;
                }
            };
            select_queue(false);
            const auto non_deferred_largest = selected_count;
            auto selected_is_deferred = false;
            auto selected_blocker_override = false;
            bool proactive_async_launch = false;
            auto owner_is_async_consumer = [&](size_t q) noexcept -> bool {
                if (q < _queue_is_async_consumer.size() &&
                    _queue_is_async_consumer[q] != 0u) {
                    return true;
                }
                const auto owner = queue_owner(q);
                return owner < _queue_is_async_consumer.size() &&
                       _queue_is_async_consumer[owner] != 0u;
            };
            auto stage_blocked_by_in_flight_async =
                [&](size_t aux_idx, uint aux_stage) noexcept -> bool {
                if (in_flight_async_count == 0u) { return false; }
                const auto &reg = _auxiliary_work[aux_idx];
                for (const auto &slot : async_slots) {
                    if (!slot.active ||
                        slot.async_domain_index >= _async_domain_names.size()) {
                        continue;
                    }
                    if (reg.work->stage_blocks_async_domain(
                            aux_stage,
                            _async_domain_names[slot.async_domain_index])) {
                        return true;
                    }
                }
                return false;
            };
            const auto min_async_overlap_count =
                (min_async_overlap_pct == 0u) ?
                    0u :
                    std::max<uint>(
                        1u,
                        static_cast<uint>(
                            (uint64_t{_active_frame_capacity} * min_async_overlap_pct) /
                            100ull));
            const uint indep_floor_n =
                (min_async_overlap_count == 0u) ?
                    0u :
                    std::min<uint>(
                        min_async_overlap_count,
                        std::max<uint>(256u, _active_frame_capacity / 32u));
            double min_t_rem_ms = std::numeric_limits<double>::infinity();
            bool any_async_domain_calibrated = false;
            if (in_flight_async_count > 0u && online_timing_active) {
                const uint64_t now_ns = steady_now_ns();
                for (const auto &slot : async_slots) {
                    if (!slot.active) { continue; }
                    const size_t dom = slot.async_domain_index;
                    if (dom < domain_async_samples.size() &&
                        domain_async_samples[dom] >= 2u) {
                        any_async_domain_calibrated = true;
                        const double elapsed_ms =
                            (slot.submit_time_ns != 0ull &&
                             now_ns >= slot.submit_time_ns) ?
                                static_cast<double>(now_ns - slot.submit_time_ns) * 1e-6 :
                                0.0;
                        const double rem_ms = std::max(
                            0.0, domain_async_est_ms[dom] - elapsed_ms);
                        min_t_rem_ms = std::min(min_t_rem_ms, rem_ms);
                    }
                }
            }
            struct OverlapEvalResult {
                bool eligible{false};
                bool calibrated{false};
                double score{0.0};
            };
            auto eval_overlap_candidate =
                [&](size_t model_idx, uint count, bool is_indep) noexcept -> OverlapEvalResult {
                if (count == 0u) { return {}; }
                const bool model_calibrated =
                    time_budget_active && any_async_domain_calibrated &&
                    model_idx < timing_models.size() &&
                    timing_models[model_idx].samples >= 2u;
                if (!is_indep) {
                    if (count < min_async_overlap_count) {
                        return {};
                    }
                    if (model_calibrated) {
                        const auto &m = timing_models[model_idx];
                        const double pred_ms = std::max(0.001, m.predict_ms(count));
                        const double eta = pred_ms / (pred_ms + t_sync_ms);
                        const double eff_rem = std::max(min_t_rem_ms, t_sync_ms);
                        const double score = std::min(pred_ms, eff_rem) * eta;
                        return {.eligible = true, .calibrated = true, .score = score};
                    }
                    return {.eligible = true,
                            .calibrated = false,
                            .score = static_cast<double>(count)};
                }
                if (count >= min_async_overlap_count) {
                    if (model_calibrated) {
                        const auto &m = timing_models[model_idx];
                        const double pred_ms = std::max(0.001, m.predict_ms(count));
                        const double eta = pred_ms / (pred_ms + t_sync_ms);
                        const double eff_rem = std::max(min_t_rem_ms, t_sync_ms);
                        const double score = std::min(pred_ms, eff_rem) * eta * 1.15;
                        return {.eligible = true, .calibrated = true, .score = score};
                    }
                    return {.eligible = true,
                            .calibrated = false,
                            .score = static_cast<double>(count) * 1.15};
                }
                if (!model_calibrated || count < indep_floor_n ||
                    min_t_rem_ms <= 0.75 * t_sync_ms) {
                    return {};
                }
                const auto &m = timing_models[model_idx];
                const double pred_ms = std::max(0.001, m.predict_ms(count));
                const double eta = pred_ms / (pred_ms + t_sync_ms);
                const double max_allowed_pred_ms =
                    2.2 * std::max(min_t_rem_ms, 0.25);
                if (pred_ms < 0.90 * t_sync_ms || pred_ms > max_allowed_pred_ms) {
                    return {.eligible = false, .calibrated = true, .score = 0.0};
                }
                const double score = std::min(pred_ms, min_t_rem_ms) * eta * 1.15;
                return {.eligible = true, .calibrated = true, .score = score};
            };
            if (_config.largest_continuation_first &&
                !deferred_blocker_nonempty &&
                proactive_async_pct > 0u &&
                eligible_async_deferred_count > 0u) {
                select_deferred_queue(true);
                if (selected < nq && selected_count != 0u &&
                    !(_extension_stages[selected - nc].handler != nullptr &&
                      _extension_stages[selected - nc].handler->prefer_deferred_entry_drain())) {
                    const size_t dom = _extension_stages[selected - nc].async_domain_index;
                    const auto proactive_async_threshold = std::max<uint>(
                        1u,
                        static_cast<uint>(
                            (uint64_t{active_count} * proactive_async_pct) / 100ull));
                    bool trigger_proactive = false;
                    if (time_budget_active && dom < domain_async_samples.size() &&
                        domain_async_samples[dom] >= 2u) {
                        const double target_async_ms = domain_async_est_ms[dom];
                        double t_indep_fuel_ms = 0.0;
                        for (size_t q = 1u; q < nq; ++q) {
                            if (q < _queue_deferred.size() && _queue_deferred[q] != 0u) {
                                continue;
                            }
                            if (is_queue_async_unavailable(q) || _host_count[q] < indep_floor_n ||
                                _host_count[q] == 0u || owner_is_async_consumer(q)) {
                                continue;
                            }
                            const size_t model_idx =
                                (q >= nc && !_extension_stages[q - nc].before_resume_chain.empty()) ?
                                    queue_owner(q) :
                                    q;
                            if (model_idx < timing_models.size() &&
                                timing_models[model_idx].samples >= 2u) {
                                t_indep_fuel_ms +=
                                    timing_models[model_idx].predict_ms(_host_count[q]);
                            } else {
                                t_indep_fuel_ms +=
                                    static_cast<double>(_host_count[q]) * 1e-5;
                            }
                        }
                        for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                            const auto &reg = _auxiliary_work[a];
                            for (uint s = 0u; s < reg.stage_count; ++s) {
                                const uint c = reg.work->stage_host_count(s);
                                if (c == 0u || c < indep_floor_n) { continue; }
                                if (dom < _async_domain_names.size() &&
                                    reg.work->stage_blocks_async_domain(
                                        s, _async_domain_names[dom])) {
                                    continue;
                                }
                                if (stage_blocked_by_in_flight_async(a, s)) {
                                    continue;
                                }
                                const size_t model_idx =
                                    static_cast<size_t>(nq) + reg.stats_offset + s;
                                if (model_idx < timing_models.size() &&
                                    timing_models[model_idx].samples >= 2u) {
                                    t_indep_fuel_ms += timing_models[model_idx].predict_ms(c);
                                } else {
                                    t_indep_fuel_ms += static_cast<double>(c) * 1e-5;
                                }
                            }
                        }
                        const uint floor_count = std::max<uint>(
                            1u, static_cast<uint>((uint64_t{active_count} * 15ull) / 100ull));
                        trigger_proactive =
                            (eligible_async_deferred_count >= proactive_async_threshold) ||
                            (eligible_async_deferred_count >= floor_count &&
                             t_indep_fuel_ms >= 0.50 * target_async_ms &&
                             t_indep_fuel_ms <= 1.35 * target_async_ms);
                    } else {
                        trigger_proactive =
                            (eligible_async_deferred_count >= proactive_async_threshold);
                    }
                    if (trigger_proactive) {
                        selected_is_deferred = true;
                        proactive_async_launch = true;
                    } else {
                        select_queue(false);
                    }
                } else {
                    select_queue(false);
                }
            }
            auto time_budget_aux_winner = _auxiliary_work.size();
            auto time_budget_aux_stage = 0u;
            bool time_budget_winner_is_main = false;
            bool found_overlap_winner = false;
            if (!proactive_async_launch && in_flight_async_count > 0u && time_budget_active) {
                bool best_calibrated = false;
                double best_score = -1.0;
                bool best_is_aux = false;
                size_t best_main_owner = nq;
                size_t best_main_selected = nq;
                uint best_main_count = 0u;
                size_t best_aux_idx = _auxiliary_work.size();
                uint best_aux_stage = 0u;
                auto accept_candidate =
                    [&](const OverlapEvalResult &ev, bool is_aux, size_t main_owner,
                        size_t main_selected, uint main_count, size_t aux_idx,
                        uint aux_stage) noexcept {
                    if (!ev.eligible) { return; }
                    if (!found_overlap_winner ||
                        (ev.calibrated && !best_calibrated) ||
                        (ev.calibrated == best_calibrated && ev.score > best_score)) {
                        found_overlap_winner = true;
                        best_calibrated = ev.calibrated;
                        best_score = ev.score;
                        best_is_aux = is_aux;
                        best_main_owner = main_owner;
                        best_main_selected = main_selected;
                        best_main_count = main_count;
                        best_aux_idx = aux_idx;
                        best_aux_stage = aux_stage;
                    }
                };
                for (size_t owner = 1u; owner < nq; ++owner) {
                    if (queue_owner(owner) != owner) { continue; }
                    if (owner < _queue_deferred.size() && _queue_deferred[owner] != 0u) {
                        continue;
                    }
                    if (is_queue_async_unavailable(owner) || logical_count[owner] == 0u) {
                        continue;
                    }
                    size_t first_q = nq;
                    for (size_t q = 1u; q < nq; ++q) {
                        if (_host_count[q] != 0u &&
                            !is_queue_async_unavailable(q) &&
                            queue_owner(q) == owner) {
                            first_q = q;
                            break;
                        }
                    }
                    if (first_q >= nq) { continue; }
                    const auto ev = eval_overlap_candidate(
                        owner, logical_count[owner], !owner_is_async_consumer(owner));
                    accept_candidate(
                        ev, false, owner, first_q, logical_count[owner],
                        _auxiliary_work.size(), 0u);
                }
                for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                    const auto &reg = _auxiliary_work[a];
                    for (uint s = 0u; s < reg.stage_count; ++s) {
                        const uint c = reg.work->stage_host_count(s);
                        if (c == 0u || stage_blocked_by_in_flight_async(a, s)) {
                            continue;
                        }
                        const size_t model_idx =
                            static_cast<size_t>(nq) + reg.stats_offset + s;
                        const auto ev = eval_overlap_candidate(model_idx, c, true);
                        accept_candidate(
                            ev, true, nq, nq, 0u, a, s);
                    }
                }
                if (found_overlap_winner) {
                    if (best_is_aux) {
                        time_budget_aux_winner = best_aux_idx;
                        time_budget_aux_stage = best_aux_stage;
                        selected = nq;
                        selected_count = 0u;
                        selected_priority = nq;
                    } else {
                        selected = best_main_selected;
                        selected_count = best_main_count;
                        selected_priority = best_main_owner;
                        time_budget_winner_is_main = true;
                    }
                }
            }
            const auto deferred_threshold =
                std::min(eligible_deferred_count / 4u, std::max(_active_frame_capacity / 8u, 1u));
            if (!proactive_async_launch && !found_overlap_winner) {
                if (_config.largest_continuation_first &&
                    eligible_deferred_count != 0u &&
                    std::max(selected_count, selected_auxiliary_count) < deferred_threshold) {
                    if (eligible_blocker_nonempty) {
                        select_blocker_queue();
                        selected_blocker_override = selected_count != 0u;
                        if (!selected_blocker_override) {
                            select_queue(false);
                        }
                    } else {
                        select_deferred_queue(false);
                        selected_is_deferred = selected_count != 0u;
                    }
                } else if (selected_count == 0u && selected_auxiliary_count == 0u) {
                    select_deferred_queue(false);
                    selected_is_deferred = selected_count != 0u;
                }
            }
            auto selected_continuation = selected_priority < nc ? selected_priority : nc;

            // Individual side stages compete with main continuations by
            // cardinality, but admission accounts for ALL live stages sharing
            // a capacity. Only if the greedy winner is a main producer do we
            // substitute its capacity owner's requested drain stage.
            auto forced_auxiliary = _auxiliary_work.size();
            auto forced_auxiliary_stage = 0u;
            auto forced_auxiliary_count = 0u;
            luisa::vector<uint8_t> aux_prepared(_auxiliary_work.size(), 0u);
            auto ensure_aux_prepared = [&](size_t a) noexcept -> uint64_t {
                auto &registered = _auxiliary_work[a];
                const auto queued = registered.work->host_count();
                const auto capacity = registered.work->capacity();
                if (aux_prepared[a] == 0u) {
                    registered.work->encode_prepare_for_admission(stream, pending_commands);
                    aux_prepared[a] = 1u;
                }
                const auto available = registered.work->host_available_slots();
                LUISA_ASSERT(registered.work->host_count() == queued &&
                                 registered.work->capacity() == capacity &&
                                 available <= capacity - queued &&
                                 (queued != 0u || available == capacity),
                             "Auxiliary '{}' admission preparation changed occupancy/capacity "
                             "or supplied invalid materializable storage.", registered.work->name());
                return std::min<uint64_t>(
                    available, capacity >= queued ? capacity - queued : 0ull);
            };
            if (selected_continuation < nc && selected_count != 0u &&
                (selected_blocker_override || time_budget_winner_is_main ||
                 selected_auxiliary_count <= selected_count)) {
                for (size_t i = 0u; i < _auxiliary_work.size(); ++i) {
                    auto &&registered = _auxiliary_work[i];
                    auto bound =
                        registered.max_emitted_per_continuation[selected_continuation];
                    if (bound == 0u) { continue; }
                    auto queued = registered.work->host_count();
                    auto capacity = registered.work->capacity();
                    const auto available = ensure_aux_prepared(i);
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
                    (time_budget_aux_winner != _auxiliary_work.size() ?
                         time_budget_aux_winner :
                         (!time_budget_winner_is_main &&
                          !selected_is_deferred && !selected_blocker_override &&
                          selected_auxiliary_count > selected_count ?
                              selected_auxiliary :
                              _auxiliary_work.size()));
            auto dispatch_auxiliary_stage =
                forced_auxiliary != _auxiliary_work.size() ?
                    forced_auxiliary_stage :
                    (time_budget_aux_winner != _auxiliary_work.size() ?
                         time_budget_aux_stage :
                         selected_auxiliary_stage);

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
            const WavefrontCoroSchedulerSnapshot scheduler_snapshot{
                .iteration = iteration_count,
                .active_frames = active_count,
                .deferred_frames = deferred_count,
                .non_deferred_frames =
                    active_count >= deferred_count ? active_count - deferred_count : 0u,
                .non_deferred_largest = non_deferred_largest,
                .deferred_blocker_frames = deferred_blocker_frames,
                .auxiliary_ready_frames = auxiliary_active_count,
                .unspawned_threads =
                    N >= dispatch_counter ? N - dispatch_counter : 0u,
                .empty_frame_slots = empty_count,
                .active_frame_capacity = _active_frame_capacity,
                .refill_threshold = refill_threshold,
                .deferred_backpressure = deferred_backpressure,
                .valid = true,
                .continuation_dispatch_seq = continuation_dispatch_seq,
                .async_consumer_dispatch_seq = async_consumer_dispatch_seq,
                .auxiliary_dispatch_seq = auxiliary_dispatch_seq,
                .in_flight_async_batches = static_cast<uint>(in_flight_async_count),
                .max_async_batch_slots = _async_slot_count};
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
            // Drain-tail compaction: once all paths have been generated, shrink a
            // sparse active prefix ([0, scan_count) when <= 25% occupied) down to
            // [0, active_count) so subsequent incremental gathers scan only live frames.
            if (_config.frame_buffer_compaction &&
                _config.incremental_continuation_counts &&
                dispatch_counter == N &&
                in_flight_async_count == 0u &&
                active_count > 0u &&
                ((scan_count >= 16384u && active_count * 4u <= scan_count) ||
                 (tail_unroll_graph_eligible &&
                  N > effective_tail_threshold &&
                  max_active_count > effective_tail_threshold &&
                  active_count <= effective_tail_threshold &&
                  scan_count > 2048u &&
                  active_count * 2u < scan_count))) {
                auto tail_holes_slot = global_count_slot;
                global_count_slot ^= 1u;
                auto gather_threads = _gather_selected_4x ?
                                          ((active_count + 1023u) / 1024u) * 256u :
                                          active_count;
                pending_commands << _gather_selected_shader(
                                        _config.thread_count, _resume_index,
                                        _global_buffer, tail_holes_slot, 0u, active_count)
                                        .dispatch(gather_threads);
                gather_scan_count += active_count;
                _host_offset[0u] = 0u;
                const auto hole_bound = std::min(active_count, compact_empty_count);
                uint verified_tail_holes = 0u;
                if (verify_queues) {
                    pending_commands << _global_buffer.view()
                                            .subview(tail_holes_slot, 1u)
                                            .copy_to(luisa::span{
                                                &verified_tail_holes, 1u});
                    sync_pending();
                    LUISA_ASSERT(
                        verified_tail_holes <= hole_bound,
                        "Drain-tail prefix hole gather violation at iteration {}: "
                        "gathered {} holes, exceeding max {}.",
                        iteration_count, verified_tail_holes, hole_bound);
                }
                auto reloc_slot = global_count_slot;
                global_count_slot ^= 1u;
                pending_commands << _compact_shader(
                                        _config.thread_count, _resume_index,
                                        _global_buffer, reloc_slot, active_count,
                                        hole_bound, scan_count)
                                        .dispatch(scan_count - active_count);
                compact_scan_count += scan_count - active_count;
                if (verify_queues) {
                    uint relocated_count = 0u;
                    pending_commands << _global_buffer.view()
                                            .subview(reloc_slot, 1u)
                                            .copy_to(luisa::span{
                                                &relocated_count, 1u});
                    sync_pending();
                    LUISA_ASSERT(
                        relocated_count == verified_tail_holes,
                        "Drain-tail compaction conservation violation at iteration {}: "
                        "relocated {} active sources into {} prefix holes.",
                        iteration_count, relocated_count, verified_tail_holes);
                }
                _used_frame_count = active_count;
                scan_count = active_count;
                compact_empty_count = 0u;
                _host_count[0u] = 0u;
                recompute_host_offsets();
            }
            if (tail_unroll_graph_eligible &&
                !online_timing_active &&
                dispatch_counter == N &&
                N > effective_tail_threshold &&
                max_active_count > effective_tail_threshold &&
                in_flight_async_count == 0u &&
                active_count > 0u &&
                active_count <= effective_tail_threshold &&
                scan_count != 0u &&
                !tail_queues.empty() &&
                !(selected_auxiliary_count > effective_tail_threshold * 2u &&
                  selected_auxiliary_count > active_count * 2u)) {
                bool all_active_queues_in_tail = true;
                for (size_t q = 1u; q < nq; ++q) {
                    if (_host_count[q] != 0u && tail_queue_used[q] == 0u) {
                        all_active_queues_in_tail = false;
                        break;
                    }
                }
                if (all_active_queues_in_tail) {
                    uint target_rounds = 0u;
                    if (tail_unroll_rounds_override != 0u) {
                        target_rounds = std::clamp<uint>(
                            tail_unroll_rounds_override, 1u, 16u);
                    } else if (tail_queues.size() <= 5u) {
                        target_rounds = (active_count <= 1024u) ? 6u : (active_count <= 4096u ? 4u : 3u);
                    } else {
                        target_rounds = (active_count <= 512u) ? 3u : 2u;
                    }
                    uint admissible_rounds = target_rounds;
                    for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                        const uint64_t per_round_req =
                            static_cast<uint64_t>(active_count) *
                            tail_aux_bound_per_round[a];
                        if (per_round_req == 0ull) { continue; }
                        const uint64_t usable = ensure_aux_prepared(a);
                        admissible_rounds = std::min<uint>(
                            admissible_rounds,
                            static_cast<uint>(usable / per_round_req));
                    }
                    if (admissible_rounds >= 2u) {
                        for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                            if (tail_aux_bound_per_round[a] == 0u) { continue; }
                            const uint64_t total_req =
                                static_cast<uint64_t>(admissible_rounds) *
                                static_cast<uint64_t>(active_count) *
                                tail_aux_bound_per_round[a];
                            _auxiliary_work[a].work->encode_prepare_for_producer(
                                stream, pending_commands,
                                static_cast<uint>(total_req));
                            aux_counts_dirty[a] = 1u;
                        }
                        luisa::vector<uint8_t> may_be_nonempty(nq, 0u);
                        for (size_t q = 1u; q < nq; ++q) {
                            if (_host_count[q] != 0u) {
                                may_be_nonempty[q] = 1u;
                            }
                        }
                        const auto gather_threads =
                            _gather_selected_4x ?
                                ((scan_count + 1023u) / 1024u) * 256u :
                                scan_count;
                        for (uint r = 0u; r < admissible_rounds; ++r) {
                            bool any_dispatched_round = false;
                            for (const size_t q : tail_queues) {
                                if (may_be_nonempty[q] == 0u) { continue; }
                                any_dispatched_round = true;
                                const size_t cont = _queue_continuation[q];
                                for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                                    const auto &reg = _auxiliary_work[a];
                                    if (cont < reg.max_emitted_per_continuation.size() &&
                                        reg.max_emitted_per_continuation[cont] != 0u) {
                                        LUISA_ASSERT(
                                            tail_aux_bound_per_round[a] != 0u &&
                                                aux_counts_dirty[a] == 1u,
                                            "Tail unroll dispatched aux-producing continuation {} "
                                            "without auxiliary admission/dirty marking.",
                                            cont);
                                    }
                                }
                                const auto count_slot = global_count_slot;
                                global_count_slot ^= 1u;
                                pending_commands << _gather_selected_shader(
                                                        _config.thread_count, _resume_index,
                                                        _global_buffer, count_slot,
                                                        static_cast<uint>(q), scan_count)
                                                        .dispatch(gather_threads);
                                gather_scan_count += scan_count;
                                pending_commands << _resume_kernels[cont](
                                                        _config.thread_count,
                                                        _resume_index.view(),
                                                        0u, 0x80000000u | count_slot,
                                                        static_cast<uint>(q), args...)
                                                        .dispatch(active_count);
                                ++continuation_dispatch_seq;
                                if (report_stats) {
                                    _last_dispatch_stats.continuations[cont].dispatch_count++;
                                }
                                may_be_nonempty[q] = 0u;
                                for (const size_t dst_q : cont_dst_queues[cont]) {
                                    may_be_nonempty[dst_q] = 1u;
                                }
                            }
                            if (!any_dispatched_round) { break; }
                        }
                        main_counts_dirty = true;
                        counts_known_on_host = false;
                        continue;
                    }
                }
            }
            if (dispatch_auxiliary == _auxiliary_work.size() &&
                !proactive_async_launch &&
                in_flight_compaction_blocking_count == 0u &&
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
                main_counts_dirty = true;
                for (size_t a = 0u; a < _auxiliary_work.size(); ++a) {
                    if (!_auxiliary_work[a].max_emitted_per_continuation.empty() &&
                        _auxiliary_work[a].max_emitted_per_continuation[0u] != 0u) {
                        aux_counts_dirty[a] = 1u;
                    }
                }
                stamp_pending_timing(0u, gen_count, in_flight_async_count > 0u);
                ++continuation_dispatch_seq;
                if (_config.frame_buffer_compaction) {
                    _used_frame_count = frame_offset + gen_count;
                }
                if (_deterministic_entry_target != 0u && !entry_emits_aux &&
                    _config.incremental_continuation_counts &&
                    _config.fused_continuation_counts && !verify_queues) {
                    _host_count[_deterministic_entry_target] += gen_count;
                    counts_known_on_host = true;
                    main_counts_dirty = false;
                }
            } else {
                const bool selected_is_eligible_async_submit =
                    dispatch_auxiliary == _auxiliary_work.size() &&
                    selected >= nc && selected < nq &&
                    _extension_stages[selected - nc].is_blocking_async &&
                    !is_queue_async_unavailable(selected);
                if (in_flight_async_count > 0u && !selected_is_eligible_async_submit) {
                    uint candidate_overlap_count = 0u;
                    size_t candidate_model_idx = total_timing_models;
                    bool candidate_indep = false;
                    bool candidate_blocked = false;
                    if (dispatch_auxiliary != _auxiliary_work.size()) {
                        const auto &reg = _auxiliary_work[dispatch_auxiliary];
                        candidate_overlap_count =
                            reg.work->stage_host_count(dispatch_auxiliary_stage);
                        candidate_model_idx =
                            static_cast<size_t>(nq) + reg.stats_offset +
                            dispatch_auxiliary_stage;
                        candidate_blocked =
                            stage_blocked_by_in_flight_async(
                                dispatch_auxiliary, dispatch_auxiliary_stage);
                        candidate_indep = !candidate_blocked;
                    } else {
                        candidate_overlap_count = selected_count;
                        candidate_model_idx =
                            (selected < nq) ?
                                ((selected >= nc &&
                                  !_extension_stages[selected - nc].before_resume_chain.empty()) ?
                                     queue_owner(selected) :
                                     selected) :
                                total_timing_models;
                        candidate_indep =
                            (selected < nq) && !owner_is_async_consumer(selected);
                    }
                    const auto gate_eval = eval_overlap_candidate(
                        candidate_model_idx, candidate_overlap_count, candidate_indep);
                    const bool gate_wait =
                        candidate_overlap_count == 0u || candidate_blocked || !gate_eval.eligible;
                    if (!gate_wait && online_timing_active) {
                        ++time_budget_overlap_dispatches;
                    }
                    if (gate_wait) {
                        if (online_timing_active) {
                            ++time_budget_overlap_waits;
                        }
                        notify_active_async_handlers(scheduler_snapshot, false);
                        if (poll_all_ready_async(scan_count, scheduler_snapshot)) {
                            continue;
                        }
                        bool launched_async_before_wait = false;
                        bool launched_from_aux = false;
                        if (in_flight_async_count < _async_slot_count &&
                            eligible_async_deferred_count != 0u &&
                            !eligible_blocker_nonempty) {
                            select_deferred_queue(true);
                            if (selected < nq && selected >= nc &&
                                _extension_stages[selected - nc].is_blocking_async &&
                                !is_queue_async_unavailable(selected) &&
                                selected_count >= std::max<uint>(min_async_overlap_count, 1u) &&
                                !(_extension_stages[selected - nc].handler != nullptr &&
                                  _extension_stages[selected - nc].handler->prefer_deferred_entry_drain())) {
                                launched_from_aux = (dispatch_auxiliary != _auxiliary_work.size());
                                selected_is_deferred = true;
                                dispatch_auxiliary = _auxiliary_work.size();
                                selected_priority = selected;
                                selected_continuation = nc;
                                launched_async_before_wait = true;
                            }
                        }
                        if (!launched_async_before_wait) {
                            wait_and_complete_one_async(
                                scan_count, scheduler_snapshot, _async_domain_names.size());
                            continue;
                        }
                        if (launched_from_aux &&
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
                    }
                }
                if (dispatch_auxiliary != _auxiliary_work.size()) {
                    auto &&registered =
                        _auxiliary_work[dispatch_auxiliary];
                    bool aux_domain_blocked = false;
                    size_t blocking_domain = domain_async_in_flight.size();
                    for (size_t d = 0u; d < domain_async_in_flight.size(); ++d) {
                        if (domain_async_in_flight[d] != 0u &&
                            registered.work->stage_blocks_async_domain(
                                dispatch_auxiliary_stage, _async_domain_names[d])) {
                            aux_domain_blocked = true;
                            blocking_domain = d;
                            break;
                        }
                    }
                    if (aux_domain_blocked) {
                        wait_and_complete_one_async(
                            scan_count, scheduler_snapshot, blocking_domain);
                        continue;
                    }
                    const uint aux_stage_count =
                        (dispatch_auxiliary < _auxiliary_work.size() &&
                         _auxiliary_work[dispatch_auxiliary].work != nullptr) ?
                            _auxiliary_work[dispatch_auxiliary].work->stage_host_count(
                                dispatch_auxiliary_stage) :
                            0u;
                    const bool aux_predicted = encode_aux_stage(
                        dispatch_auxiliary, dispatch_auxiliary_stage, scheduler_snapshot);
                    if (aux_predicted) {
                        counts_known_on_host = true;
                        continue;
                    }
                    if (batch_queues_enabled &&
                        _config.largest_continuation_first &&
                        _config.incremental_continuation_counts &&
                        _config.fused_continuation_counts &&
                        in_flight_async_count == 0u &&
                        !online_timing_active &&
                        scan_count != 0u) {
                        const uint batch_min_count = compute_batch_min_count(scan_count);
                        bool any_non_collect_nonempty = false;
                        for (size_t q = 1u; q < nq; ++q) {
                            if (_host_count[q] == 0u) { continue; }
                            if (q < _queue_deferred.size() && _queue_deferred[q] != 0u) { continue; }
                            if (!(q < _collect_at.size() && _collect_at[q])) {
                                any_non_collect_nonempty = true;
                                break;
                            }
                        }
                        luisa::vector<size_t> piggy_cands;
                        for (size_t q = 1u; q < nq; ++q) {
                            if (!queue_is_batchable_main(q)) { continue; }
                            if (aux_stage_count >= batch_min_count &&
                                _host_count[q] < batch_min_count) {
                                continue;
                            }
                            const auto cont = _queue_continuation[q];
                            bool emits_aux = false;
                            for (const auto &reg : _auxiliary_work) {
                                if (cont < reg.max_emitted_per_continuation.size() &&
                                    reg.max_emitted_per_continuation[cont] != 0u) {
                                    emits_aux = true;
                                    break;
                                }
                            }
                            if (emits_aux) { continue; }
                            if (dispatch_counter < N && should_refill &&
                                q < _refill_at.size() && _refill_at[q]) {
                                continue;
                            }
                            if (any_non_collect_nonempty &&
                                q < _collect_at.size() && _collect_at[q]) {
                                continue;
                            }
                            piggy_cands.emplace_back(q);
                        }
                        std::sort(piggy_cands.begin(), piggy_cands.end(),
                                  [&](size_t a, size_t b) noexcept {
                                      if (_host_count[a] != _host_count[b]) {
                                          return _host_count[a] > _host_count[b];
                                      }
                                      return queue_owner(a) < queue_owner(b);
                                  });
                        luisa::vector<size_t> piggy_batch;
                        luisa::vector<size_t> blocked;
                        for (size_t q : piggy_cands) {
                            const auto cont_q = _queue_continuation[q];
                            bool is_blocked = false;
                            for (size_t b : blocked) {
                                if (can_produce_into(_queue_continuation[b], cont_q)) {
                                    is_blocked = true;
                                    break;
                                }
                            }
                            if (is_blocked) {
                                blocked.emplace_back(q);
                                continue;
                            }
                            piggy_batch.emplace_back(q);
                            blocked.emplace_back(q);
                        }
                        bool all_piggy_sinks = !piggy_batch.empty();
                        for (size_t q : piggy_batch) {
                            if (!encode_single_member_main_queue(
                                    q, scan_count, scheduler_snapshot)) {
                                all_piggy_sinks = false;
                            }
                        }
                        if (!piggy_batch.empty() && all_piggy_sinks) {
                            for (size_t q : piggy_batch) {
                                _host_count[q] = 0u;
                            }
                            main_counts_dirty = false;
                        }
                    }
                    continue;
                }
                if (selected_count == 0u) {
                    LUISA_ASSERT(in_flight_async_count > 0u,
                                 "Wavefront scheduler has no eligible ready queue at iteration {} "
                                 "(active={}, aux={}).",
                                 iteration_count, active_count, auxiliary_active_count);
                    wait_and_complete_one_async(
                        scan_count, scheduler_snapshot, _async_domain_names.size());
                    continue;
                }
                const bool try_main_batch =
                    batch_queues_enabled &&
                    _config.largest_continuation_first &&
                    _config.incremental_continuation_counts &&
                    _config.fused_continuation_counts &&
                    in_flight_async_count == 0u &&
                    !online_timing_active &&
                    !selected_is_deferred &&
                    !selected_blocker_override &&
                    !proactive_async_launch &&
                    scan_count != 0u &&
                    queue_is_batchable_main(selected);
                if (try_main_batch) {
                    const size_t num_aux = _auxiliary_work.size();
                    luisa::vector<uint64_t> rem_cap(num_aux, 0ull);
                    luisa::vector<uint64_t> rem_avail(num_aux, 0ull);
                    luisa::vector<uint64_t> batch_req(num_aux, 0ull);
                    for (size_t a = 0u; a < num_aux; ++a) {
                        const auto &reg = _auxiliary_work[a];
                        const uint64_t cap = reg.work->capacity();
                        const uint64_t queued = reg.work->host_count();
                        rem_cap[a] = cap >= queued ? cap - queued : 0ull;
                        rem_avail[a] = aux_prepared[a] ?
                                           static_cast<uint64_t>(reg.work->host_available_slots()) :
                                           0ull;
                        if (selected_continuation < reg.max_emitted_per_continuation.size()) {
                            const uint64_t bound = reg.max_emitted_per_continuation[selected_continuation];
                            if (bound != 0u) {
                                const uint64_t req = static_cast<uint64_t>(selected_count) * bound;
                                batch_req[a] += req;
                                rem_cap[a] = rem_cap[a] >= req ? rem_cap[a] - req : 0ull;
                                rem_avail[a] = rem_avail[a] >= req ? rem_avail[a] - req : 0ull;
                            }
                        }
                    }
                    luisa::vector<size_t> batch{selected};
                    const bool selected_is_collect_or_refill =
                        (selected < _collect_at.size() && _collect_at[selected]) ||
                        (dispatch_counter < N && selected < _refill_at.size() && _refill_at[selected]);
                    if (!selected_is_collect_or_refill) {
                        const uint sel_count = _host_count[selected];
                        const uint batch_min_count = compute_batch_min_count(scan_count);
                        luisa::vector<size_t> main_cands;
                        for (size_t q = 1u; q < nq; ++q) {
                            if (q == selected) { continue; }
                            if (!queue_is_batchable_main(q)) { continue; }
                            if (sel_count >= batch_min_count &&
                                _host_count[q] < batch_min_count) {
                                continue;
                            }
                            if (q < _collect_at.size() && _collect_at[q]) { continue; }
                            if (dispatch_counter < N && should_refill &&
                                q < _refill_at.size() && _refill_at[q]) {
                                continue;
                            }
                            main_cands.emplace_back(q);
                        }
                        std::sort(main_cands.begin(), main_cands.end(),
                                  [&](size_t a, size_t b) noexcept {
                                      if (_host_count[a] != _host_count[b]) {
                                          return _host_count[a] > _host_count[b];
                                      }
                                      return queue_owner(a) < queue_owner(b);
                                  });
                        luisa::vector<size_t> blocked{selected};
                        for (size_t q : main_cands) {
                            const auto cont_q = _queue_continuation[q];
                            bool is_blocked = false;
                            for (size_t b : blocked) {
                                if (can_produce_into(_queue_continuation[b], cont_q)) {
                                    is_blocked = true;
                                    break;
                                }
                            }
                            if (is_blocked) {
                                blocked.emplace_back(q);
                                continue;
                            }
                            bool admissible = true;
                            for (size_t a = 0u; a < num_aux; ++a) {
                                auto &reg = _auxiliary_work[a];
                                const uint64_t bound =
                                    cont_q < reg.max_emitted_per_continuation.size() ?
                                        reg.max_emitted_per_continuation[cont_q] :
                                        0ull;
                                if (bound == 0u) { continue; }
                                const uint64_t req = static_cast<uint64_t>(_host_count[q]) * bound;
                                if (req > rem_cap[a]) {
                                    admissible = false;
                                    break;
                                }
                                if (aux_prepared[a] == 0u) {
                                    const auto queued = reg.work->host_count();
                                    const auto capacity = reg.work->capacity();
                                    reg.work->encode_prepare_for_admission(stream, pending_commands);
                                    const auto available = reg.work->host_available_slots();
                                    LUISA_ASSERT(reg.work->host_count() == queued &&
                                                     reg.work->capacity() == capacity &&
                                                     available <= capacity - queued &&
                                                     (queued != 0u || available == capacity),
                                                 "Auxiliary '{}' admission preparation changed occupancy/capacity "
                                                 "or supplied invalid materializable storage.",
                                                 reg.work->name());
                                    rem_avail[a] = available >= batch_req[a] ? available - batch_req[a] : 0ull;
                                    aux_prepared[a] = 1u;
                                }
                                if (req > rem_avail[a]) {
                                    admissible = false;
                                    break;
                                }
                            }
                            if (!admissible) {
                                blocked.emplace_back(q);
                                continue;
                            }
                            for (size_t a = 0u; a < num_aux; ++a) {
                                const auto &reg = _auxiliary_work[a];
                                const uint64_t bound =
                                    cont_q < reg.max_emitted_per_continuation.size() ?
                                        reg.max_emitted_per_continuation[cont_q] :
                                        0ull;
                                if (bound == 0u) { continue; }
                                const uint64_t req = static_cast<uint64_t>(_host_count[q]) * bound;
                                batch_req[a] += req;
                                rem_cap[a] -= req;
                                rem_avail[a] -= req;
                            }
                            batch.emplace_back(q);
                            blocked.emplace_back(q);
                        }
                    }
                    for (size_t a = 0u; a < num_aux; ++a) {
                        if (batch_req[a] != 0u) {
                            _auxiliary_work[a].work->encode_prepare_for_producer(
                                stream, pending_commands, static_cast<uint>(batch_req[a]));
                        }
                    }
                    bool all_batch_sinks = true;
                    for (size_t q : batch) {
                        if (!encode_single_member_main_queue(
                                q, scan_count, scheduler_snapshot)) {
                            all_batch_sinks = false;
                        }
                    }
                    if (all_batch_sinks) {
                        for (size_t q : batch) {
                            _host_count[q] = 0u;
                        }
                        counts_known_on_host = true;
                        main_counts_dirty = false;
                    } else {
                        const bool can_post_batch_chain =
                            post_batch_chain_enabled &&
                            _tail_unroll_supported &&
                            !verify_queues &&
                            in_flight_async_count == 0u &&
                            !online_timing_active &&
                            (dispatch_counter == N || !should_refill);
                        if (can_post_batch_chain) {
                            luisa::vector<uint8_t> in_batch(nq, 0u);
                            for (const size_t bq : batch) {
                                if (bq < nq) {
                                    in_batch[bq] = 1u;
                                }
                            }
                            struct FollowupCandidate {
                                size_t queue_index;
                                uint bound;
                            };
                            luisa::vector<FollowupCandidate> followup_cands;
                            for (size_t fq = 1u; fq < nc; ++fq) {
                                if (in_batch[fq] != 0u) { continue; }
                                if (!_resume_kernels[fq]) { continue; }
                                if (fq < _queue_deferred.size() && _queue_deferred[fq] != 0u) {
                                    continue;
                                }
                                if (_has_hint_sort && fq < _have_hint.size() && _have_hint[fq]) {
                                    continue;
                                }
                                const bool is_hub =
                                    (fq < _refill_at.size() && _refill_at[fq]) ||
                                    (fq < _collect_at.size() && _collect_at[fq]);
                                const bool is_term =
                                    fq < _is_terminal_continuation.size() &&
                                    _is_terminal_continuation[fq];
                                if (!is_hub && !is_term) { continue; }
                                if (!((logical_members[fq] == 0u) ||
                                      (logical_members[fq] == 1u && _host_count[fq] != 0u))) {
                                    continue;
                                }
                                bool fq_emits_aux = false;
                                for (size_t a = 0u; a < num_aux; ++a) {
                                    const auto &reg = _auxiliary_work[a];
                                    if (fq < reg.max_emitted_per_continuation.size() &&
                                        reg.max_emitted_per_continuation[fq] != 0u) {
                                        fq_emits_aux = true;
                                        break;
                                    }
                                }
                                if (fq_emits_aux) { continue; }
                                if (fq < _collect_at.size() && _collect_at[fq]) {
                                    bool blocked_by_unbatched = false;
                                    for (size_t uq = 1u; uq < nq; ++uq) {
                                        if (_host_count[uq] == 0u || in_batch[uq] != 0u) {
                                            continue;
                                        }
                                        if (uq < _collect_at.size() && _collect_at[uq]) {
                                            continue;
                                        }
                                        const size_t ucont = _queue_continuation[uq];
                                        if (ucont < nc && can_produce_into(ucont, fq)) {
                                            blocked_by_unbatched = true;
                                            break;
                                        }
                                    }
                                    if (blocked_by_unbatched) { continue; }
                                }
                                uint64_t produced_bound = 0ull;
                                for (const size_t bq : batch) {
                                    const size_t bcont = _queue_continuation[bq];
                                    if (bcont < cont_dst_queues.size()) {
                                        const auto &dsts = cont_dst_queues[bcont];
                                        if (std::find(dsts.begin(), dsts.end(), fq) != dsts.end()) {
                                            produced_bound += _host_count[bq];
                                        }
                                    }
                                }
                                if (produced_bound == 0ull) { continue; }
                                const uint fq_bound = static_cast<uint>(std::min<uint64_t>({
                                    static_cast<uint64_t>(scan_count),
                                    static_cast<uint64_t>(active_count),
                                    static_cast<uint64_t>(_host_count[fq]) + produced_bound}));
                                if (is_hub) {
                                    if (fq_bound < 2048u) { continue; }
                                } else {
                                    if (produced_bound < std::max<uint64_t>(
                                                             4096u,
                                                             static_cast<uint64_t>(scan_count) / 16u) ||
                                        fq_bound < 4096u) {
                                        continue;
                                    }
                                }
                                followup_cands.push_back(FollowupCandidate{fq, fq_bound});
                            }
                            std::sort(followup_cands.begin(), followup_cands.end(),
                                      [](const FollowupCandidate &a, const FollowupCandidate &b) noexcept {
                                          if (a.bound != b.bound) {
                                              return a.bound > b.bound;
                                          }
                                          return a.queue_index < b.queue_index;
                                      });
                            luisa::vector<FollowupCandidate> followup_queues;
                            for (const auto &cand : followup_cands) {
                                bool conflicts = false;
                                for (const auto &kept : followup_queues) {
                                    if (can_produce_into(kept.queue_index, cand.queue_index) ||
                                        can_produce_into(cand.queue_index, kept.queue_index)) {
                                        conflicts = true;
                                        break;
                                    }
                                }
                                if (!conflicts) {
                                    followup_queues.push_back(cand);
                                }
                            }
                            if (!followup_queues.empty()) {
                                const auto gather_threads =
                                    _gather_selected_4x ?
                                        ((scan_count + 1023u) / 1024u) * 256u :
                                        scan_count;
                                for (const auto &f : followup_queues) {
                                    const auto count_slot = global_count_slot;
                                    global_count_slot ^= 1u;
                                    pending_commands << _gather_selected_shader(
                                                            _config.thread_count, _resume_index,
                                                            _global_buffer, count_slot,
                                                            static_cast<uint>(f.queue_index), scan_count)
                                                            .dispatch(gather_threads);
                                    gather_scan_count += scan_count;
                                    pending_commands << _resume_kernels[f.queue_index](
                                                            _config.thread_count,
                                                            _resume_index.view(),
                                                            0u, 0x80000000u | count_slot,
                                                            static_cast<uint>(f.queue_index), args...)
                                                            .dispatch(f.bound);
                                    ++continuation_dispatch_seq;
                                    if (report_stats) {
                                        _last_dispatch_stats.continuations[f.queue_index].dispatch_count++;
                                    }
                                }
                            }
                        }
                        if (selected_auxiliary < num_aux &&
                            selected_auxiliary_count != 0u &&
                            batch_req[selected_auxiliary] == 0u) {
                            static_cast<void>(encode_aux_stage(
                                selected_auxiliary, selected_auxiliary_stage, scheduler_snapshot));
                        }
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
                            luisa::vector<uint> ends(ntok);
                            pending_commands << _resume_offset.copy_to(luisa::span{ends});
                            sync_pending();
                            for (size_t i = 0u; i < ntok; ++i) {
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
                                        first_handler.gathers_selected_frames_for(
                                            selected_count, scan_count);
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
                bool async_submitted = false;
                bool used_sync_filtered = false;
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
                    if (is_queue_async_unavailable(i)) {
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
                    bool handled_blocking_async = false;
                    if (registered.is_blocking_async) {
                        size_t picked_slot = async_slots.size();
                        for (size_t k = 0u; k < async_slots.size(); ++k) {
                            const size_t cand = (next_slot_cursor + k) % async_slots.size();
                            if (!async_slots[cand].active) {
                                picked_slot = cand;
                                break;
                            }
                        }
                        LUISA_ASSERT(picked_slot < async_slots.size(),
                                     "No free async batch slot available while queue {} appeared eligible.",
                                     i);
                        const WavefrontCoroAsyncBatchTicket ticket{
                            .slot_index = static_cast<uint>(picked_slot),
                            .generation = ++next_async_generation};
                        CommandList *handler_commands = nullptr;
                        if (registered.handler->supports_command_list()) {
                            handler_commands = &pending_commands;
                        } else {
                            flush_pending();
                        }
                        bool accepted = false;
                        {
                            detail::WavefrontCoroSchedulerSnapshotScope snapshot_scope{scheduler_snapshot};
                            accepted = registered.handler->submit_async(WavefrontCoroExtensionDispatchContext{
                                .stream = stream,
                                .frame_buffer = _frame_buffer.view(),
                                .frame_indices = indices,
                                .frame_count = count,
                                .frame_capacity = _config.thread_count,
                                .logical_dispatch_size = dispatch_size,
                                .stage = registered.stage,
                                .command_list = handler_commands,
                                .source_queue_token = static_cast<uint>(i),
                                .scan_count = scan_count,
                                .ungathered_scan = false,
                                .scheduler_snapshot = scheduler_snapshot,
                                .async_ticket = ticket});
                        }
                        if (accepted) {
                            next_slot_cursor = (picked_slot + 1u) % async_slots.size();
                            const uint parked_tok = static_cast<uint>(nq + picked_slot);
                            pending_commands << _advance_extension_stage_shader(
                                                    _config.thread_count, indices,
                                                    0u, count, static_cast<uint>(i),
                                                    parked_tok)
                                                    .dispatch(advance_dispatch_threads(count));
                            if (_config.incremental_continuation_counts &&
                                !_config.fused_continuation_counts) {
                                pending_commands << _publish_resumed_count_shader(
                                                        _config.thread_count, indices, 0u,
                                                        _resume_count,
                                                        static_cast<uint>(i), count)
                                                        .dispatch(count);
                            }
                            const bool blocks_compaction =
                                registered.modifies_frame &&
                                !registered.handler->supports_compaction_while_async();
                            const bool was_deferred =
                                i < _queue_deferred.size() && _queue_deferred[i] != 0u;
                            const uint def_target =
                                (was_deferred && i < _queue_deferred_target.size()) ?
                                    _queue_deferred_target[i] :
                                    static_cast<uint>(nc);
                            if (selected_is_deferred && i == selected &&
                                i < _queue_deferred_target.size()) {
                                last_deferred_target = _queue_deferred_target[i];
                            }
                            async_slots[picked_slot] = AsyncBatchSlot{
                                .active = true,
                                .generation = ticket.generation,
                                .submit_seq = ++next_async_submit_seq,
                                .submit_time_ns = online_timing_active ? steady_now_ns() : 0ull,
                                .stage_index = stage_index,
                                .parked_token = parked_tok,
                                .source_queue = static_cast<uint>(i),
                                .next_queue = registered.next_queue,
                                .frame_count = count,
                                .deferred_target = def_target,
                                .was_deferred = was_deferred,
                                .blocks_compaction = blocks_compaction,
                                .async_domain_index = registered.async_domain_index};
                            if (online_timing_active) {
                                pending_timed_slot = total_timing_models;
                                pending_timed_count = 0u;
                                pending_timed_dispatches = 0u;
                                pending_timed_was_overlap = false;
                            }
                            if (registered.async_domain_index < domain_async_in_flight.size()) {
                                ++domain_async_in_flight[registered.async_domain_index];
                            }
                            if (blocks_compaction) {
                                ++in_flight_compaction_blocking_count;
                            }
                            ++in_flight_async_count;
                            _host_count[i] = 0u;
                            _host_count[parked_tok] = count;
                            if (_config.incremental_continuation_counts) {
                                recompute_host_offsets();
                            }
                            if (_config.largest_continuation_first &&
                                _config.incremental_continuation_counts &&
                                !verify_queues) {
                                counts_known_on_host =
                                    _config.fused_continuation_counts;
                            }
                            if (!counts_known_on_host) {
                                main_counts_dirty = true;
                            }
                            async_submitted = true;
                            if (_config.largest_continuation_first) {
                                break;
                            }
                            continue;
                        }
                        handled_blocking_async = true;
                        if (report_stats) {
                            auto &stat = _last_dispatch_stats.extensions[stage_index];
                            stat.dispatch_count++;
                            stat.executed_count += count;
                            stat.peak_queued_count = std::max(stat.peak_queued_count, count);
                        }
                        _last_dispatch_stats.extension_count += count;
                    }
                    if (!handled_blocking_async) {
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
                            {
                                detail::WavefrontCoroSchedulerSnapshotScope snapshot_scope{scheduler_snapshot};
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
                                    .ungathered_scan = is_ungathered,
                                    .scheduler_snapshot = scheduler_snapshot});
                            }
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
                                main_counts_dirty = false;
                            }
                            continue;
                        }
                        dispatch_handler(stage_index);
                        if (registered.before_resume_chain.empty()) {
                            stamp_pending_timing(i, count, in_flight_async_count > 0u);
                        }
                    }
                    if (selected_is_deferred && i == selected &&
                        i < _queue_deferred_target.size()) {
                        last_deferred_target = _queue_deferred_target[i];
                    }
                    bool use_sync_filtered = false;
                    if (count != 0u &&
                        registered.handler != nullptr &&
                        registered.handler->supports_filtered_advance() &&
                        static_cast<bool>(_advance_extension_stage_filtered_shader)) {
                        if (filtered_zero_streak[stage_index] >= 2u) {
                            filtered_zero_streak[stage_index] = 0u;
                        } else {
                            use_sync_filtered = true;
                        }
                    }
                    if (use_sync_filtered) {
                        registered.handler->encode_advance_filter(WavefrontCoroAdvanceFilterContext{
                            .stream = stream,
                            .command_list = pending_commands,
                            .frame_buffer = _frame_buffer.view(),
                            .frame_indices = indices,
                            .ready_flags = _advance_ready_flags.view(0u, count),
                            .frame_count = count,
                            .frame_capacity = _config.thread_count,
                            .stage = registered.stage});
                        pending_commands << _advance_extension_stage_filtered_shader(
                                                _config.thread_count, indices,
                                                0u, count, static_cast<uint>(i),
                                                registered.next_queue, static_cast<uint>(i),
                                                _advance_ready_flags)
                                                .dispatch(advance_dispatch_threads(count));
                    } else {
                        pending_commands << _advance_extension_stage_shader(
                                                _config.thread_count, indices,
                                                0u, count, static_cast<uint>(i),
                                                registered.next_queue)
                                                .dispatch(advance_dispatch_threads(count));
                    }
                    if (_config.incremental_continuation_counts &&
                        !_config.fused_continuation_counts) {
                        pending_commands << _publish_resumed_count_shader(
                                                _config.thread_count, indices, 0u,
                                                _resume_count,
                                                static_cast<uint>(i), count)
                                                .dispatch(count);
                    }
                    if (use_sync_filtered) {
                        pending_filtered_stage = stage_index;
                        pending_filtered_next_queue = registered.next_queue;
                        pending_filtered_prev_next_count =
                            registered.next_queue < nq ?
                                _host_count[registered.next_queue] :
                                0u;
                        counts_known_on_host = false;
                        main_counts_dirty = true;
                        used_sync_filtered = true;
                        break;
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
                    } else {
                        main_counts_dirty = true;
                    }
                }
                if (used_sync_filtered ||
                    (async_submitted && _config.largest_continuation_first)) {
                    continue;
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
                        main_counts_dirty = false;
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
            _last_dispatch_stats.online_timing_active = online_timing_active;
            _last_dispatch_stats.online_timing_hw_timestamps = online_hw_timestamps;
            _last_dispatch_stats.online_sync_bubble_ms =
                online_timing_active ? t_sync_ms : 0.0;
            _last_dispatch_stats.time_budget_overlap_dispatches =
                time_budget_overlap_dispatches;
            _last_dispatch_stats.time_budget_overlap_waits =
                time_budget_overlap_waits;
            LUISA_INFO("Wavefront stats: iterations={} generated={} resumed={} extensions={} gather_scan={} compact_scan={} max_scan={} max_active={} elapsed_ms={:.3f}",
                       iteration_count, generated_count, resumed_count,
                       _last_dispatch_stats.extension_count,
                       gather_scan_count, compact_scan_count,
                       max_scan_count, max_active_count,
                       _last_dispatch_stats.elapsed_ms);
            if (online_timing_active) {
                uint32_t calibrated_models = 0u;
                for (const auto &tm : timing_models) {
                    if (tm.samples >= 2u) {
                        ++calibrated_models;
                    }
                }
                const double dom0_est_ms =
                    domain_async_est_ms.empty() ? 0.0 : domain_async_est_ms[0u];
                LUISA_INFO("Wavefront online timing: hw_ts={} time_budget={} sync_bubble_ms={:.3f} overlap_dispatches={} overlap_waits={} calibrated_models={} hw_ts_cbs={} dom0_est_ms={:.3f}",
                           online_hw_timestamps ? 1 : 0,
                           time_budget_active ? 1 : 0,
                           t_sync_ms,
                           time_budget_overlap_dispatches,
                           time_budget_overlap_waits,
                           calibrated_models,
                           hw_ts_cbs,
                           dom0_est_ms);
            }
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
