// OCR engine -- decode KV cache + device-visible decode metadata (Phase 3).
#include "ocr/runtime/decode_cache.h"

#include <cstring>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/runtime/command_list.h>

#include "ocr/quantization.h"
#include "ocr/runtime/memory_budget.h"

namespace luisa::ocr {

[[nodiscard]] Status KvCacheConfig::validate() const {
    if (num_layers == 0u || num_kv_heads == 0u || head_dim == 0u || max_seq_len == 0u || batch == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache geometry fields must all be nonzero", "config");
    }
    if (dtype_bytes != 4u) {
        return Status::error(StatusCode::unsupported,
                             "Phase 3 contiguous KV storage is FP32 (dtype_bytes == 4)", "dtype_bytes");
    }
    return Status::success();
}

[[nodiscard]] Result<uint64_t> KvCacheConfig::total_bytes() const {
    auto s = validate();
    if (!s.ok()) { return Result<uint64_t>::failure(s); }
    return plan_kv_cache_bytes(num_layers, batch, max_seq_len, num_kv_heads, head_dim, dtype_bytes);
}

[[nodiscard]] Result<uint64_t> KvCacheConfig::layer_elements() const {
    auto s = validate();
    if (!s.ok()) { return Result<uint64_t>::failure(s); }
    uint64_t per_seq = 0, total = 0;
    if (!checked_mul(max_seq_len, num_kv_heads, per_seq) ||
        !checked_mul(per_seq, head_dim, per_seq) ||
        !checked_mul(per_seq, batch, total)) {
        return Result<uint64_t>::failure(StatusCode::invalid_argument,
                                         "KV layer element count overflows uint64", "layer_elements");
    }
    return Result<uint64_t>::success(total);
}

// ---------------------------------------------------------------------------
// DecodeState
// ---------------------------------------------------------------------------

[[nodiscard]] Status DecodeState::create(compute::Device &device, compute::Stream &stream, uint32_t batch) {
    if (batch == 0u) {
        return Status::error(StatusCode::invalid_argument, "decode state batch must be nonzero", "batch");
    }
    _kv_len = device.create_buffer<uint32_t>(batch);
    _positions = device.create_buffer<int32_t>(batch);
    _step = device.create_buffer<uint32_t>(batch);
    _state = device.create_buffer<uint32_t>(batch);
    if (!_kv_len || !_positions || !_step || !_state) {
        _kv_len = {};
        _positions = {};
        _step = {};
        _state = {};
        return Status::error(StatusCode::out_of_memory, "failed to allocate decode metadata buffers", "batch");
    }
    _batch = batch;
    // Synchronous zero initialization: an unbound/released sequence must never
    // be attended. The staging vectors live until the drain below completes.
    const luisa::vector<uint32_t> zeros(batch, 0u);
    const luisa::vector<int32_t> pos_zeros(batch, 0);
    stream << _kv_len.copy_from(luisa::span<const uint32_t>{zeros})
           << _positions.copy_from(luisa::span<const int32_t>{pos_zeros})
           << _step.copy_from(luisa::span<const uint32_t>{zeros})
           << _state.copy_from(luisa::span<const uint32_t>{zeros})
           << compute::synchronize();
    return Status::success();
}

[[nodiscard]] Status DecodeState::record(compute::CommandList &list,
                                         luisa::span<const uint32_t> kv_len,
                                         luisa::span<const int32_t> positions,
                                         luisa::span<const uint32_t> step,
                                         luisa::span<const uint32_t> state) const {
    if (!valid()) {
        return Status::error(StatusCode::invalid_argument, "decode state is not created", "state");
    }
    if (kv_len.size() != _batch || positions.size() != _batch ||
        step.size() != _batch || state.size() != _batch) {
        return Status::error(StatusCode::invalid_argument,
                             "decode metadata snapshot size must equal batch", "batch");
    }
    // Heap-owned staging retained through command completion: BufferUpload
    // borrows the host pointer, so the snapshot must outlive EXECUTION, not
    // just this call. A completion callback owns and releases it. This makes
    // multiple records into one CommandList (and stack-local caller spans)
    // safe -- each record owns a distinct staging block.
    struct Staging {
        luisa::vector<uint32_t> kv_len;
        luisa::vector<int32_t> positions;
        luisa::vector<uint32_t> step;
        luisa::vector<uint32_t> state;
    };
    auto staging = luisa::make_unique<Staging>();
    staging->kv_len.assign(kv_len.begin(), kv_len.end());
    staging->positions.assign(positions.begin(), positions.end());
    staging->step.assign(step.begin(), step.end());
    staging->state.assign(state.begin(), state.end());
    auto *raw = staging.get();
    list << _kv_len.copy_from(luisa::span<const uint32_t>{raw->kv_len})
         << _positions.copy_from(luisa::span<const int32_t>{raw->positions})
         << _step.copy_from(luisa::span<const uint32_t>{raw->step})
         << _state.copy_from(luisa::span<const uint32_t>{raw->state});
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

[[nodiscard]] Status DecodeState::record_sequence(compute::CommandList &list, uint32_t index,
                                                  uint32_t kv_len, int32_t position,
                                                  DecodeSeqState state) const {
    if (!valid() || index >= _batch) {
        return Status::error(StatusCode::invalid_argument, "decode sequence index out of range", "index");
    }
    // Single-element staging: BufferUploadCommand borrows the host pointer,
    // so the staging block must outlive command EXECUTION, not just encoding.
    // A command-list callback runs after the list's commands complete; it
    // owns the staging and releases it there (no per-step host allocation
    // beyond this 12-byte boundary record).
    struct Staging {
        uint32_t kv_len;
        int32_t position;
        uint32_t state;
    };
    auto staging = luisa::make_unique<Staging>(Staging{kv_len, position, static_cast<uint32_t>(state)});
    auto *raw = staging.get();
    list << _kv_len.view(index, 1).copy_from(luisa::span<const uint32_t>{&raw->kv_len, 1})
         << _positions.view(index, 1).copy_from(luisa::span<const int32_t>{&raw->position, 1})
         << _state.view(index, 1).copy_from(luisa::span<const uint32_t>{&raw->state, 1});
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

[[nodiscard]] uint64_t DecodeState::device_bytes() const noexcept {
    if (!valid()) { return 0u; }
    return _kv_len.size_bytes() + _positions.size_bytes() + _step.size_bytes() + _state.size_bytes();
}

[[nodiscard]] Status DecodeState::download(compute::Stream &stream,
                                           luisa::vector<uint32_t> &kv_len,
                                           luisa::vector<int32_t> &positions,
                                           luisa::vector<uint32_t> &step,
                                           luisa::vector<uint32_t> &state) const {
    if (!valid()) {
        return Status::error(StatusCode::invalid_argument, "decode state is not created", "state");
    }
    kv_len.resize(_batch);
    positions.resize(_batch);
    step.resize(_batch);
    state.resize(_batch);
    stream << _kv_len.copy_to(luisa::span<uint32_t>{kv_len})
           << _positions.copy_to(luisa::span<int32_t>{positions})
           << _step.copy_to(luisa::span<uint32_t>{step})
           << _state.copy_to(luisa::span<uint32_t>{state});
    return Status::success();
}

// ---------------------------------------------------------------------------
// ContiguousKvCache
// ---------------------------------------------------------------------------

[[nodiscard]] Status ContiguousKvCache::create(compute::Device &device, compute::Stream &stream,
                                               const KvCacheConfig &config, MemoryBudget *budget) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    auto bytes = config.total_bytes();
    if (!bytes.ok()) { return bytes.status; }
    auto elems = config.layer_elements();
    if (!elems.ok()) { return elems.status; }
    const uint64_t per_buffer_bytes = *bytes.value / 2u;// K and V are equal
    // DecodeState metadata: 4 buffers (kv_len/positions/step/state) x batch x
    // 4 bytes. Included in BOTH the preflight and the recorded requirement so
    // the budget accounts for the full resident footprint.
    const uint64_t metadata_bytes = 4ull * config.batch * 4ull;
    uint64_t total_required = 0u;
    if (!checked_add(*bytes.value, metadata_bytes, total_required)) {
        return Status::error(StatusCode::out_of_memory, "KV cache byte requirement overflows", "capacity");
    }
    if (budget != nullptr) {
        auto pre = budget->preflight_additional_device(total_required);
        if (!pre.ok()) { return pre; }
    }
    // Element counts must fit size_t for buffer creation.
    const uint64_t total_elems = *elems.value * config.num_layers;
    if (total_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return Status::error(StatusCode::out_of_memory, "KV element count exceeds size_t", "capacity");
    }
    auto k = device.create_buffer<float>(static_cast<size_t>(total_elems));
    auto v = device.create_buffer<float>(static_cast<size_t>(total_elems));
    if (!k || !v) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("failed to allocate contiguous KV cache ({} bytes per K/V buffer)",
                                           per_buffer_bytes),
                             "capacity");
    }
    // Synchronous zero-init of the device metadata (drains the stream).
    auto state_status = _state.create(device, stream, config.batch);
    if (!state_status.ok()) { return state_status; }
    _config = config;
    _k = std::move(k);
    _v = std::move(v);
    _layer_stride = *elems.value;
    _lengths.assign(config.batch, 0u);
    if (budget != nullptr) {
        auto s = budget->require("decode_cache", MemoryCategory::kv_cache, device_bytes());
        if (!s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] uint32_t ContiguousKvCache::length(uint32_t index) const noexcept {
    return index < _lengths.size() ? _lengths[index] : 0u;
}

void ContiguousKvCache::release(uint32_t index) noexcept {
    if (index < _lengths.size()) { _lengths[index] = 0u; }
    // Host capacity is released HERE so a new sequence can bind immediately
    // (no pool churn, no per-step realloc). The device-side finished state is
    // recorded separately via record_finished() at the completion boundary
    // (the controller does this); release() itself issues no device commands.
}

[[nodiscard]] uint64_t ContiguousKvCache::device_bytes() const noexcept {
    if (!valid()) { return 0u; }
    return _k.size_bytes() + _v.size_bytes() + _state.device_bytes();
}

[[nodiscard]] Status ContiguousKvCache::bind_sequence(uint32_t index, uint32_t prompt_len) {
    if (!valid() || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "KV sequence index out of range", "index");
    }
    if (_lengths[index] != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "KV sequence slot is occupied; release() it first", "index");
    }
    if (prompt_len > _config.max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("prompt length {} exceeds KV capacity {}",
                                           prompt_len, _config.max_seq_len),
                             "prompt_len");
    }
    _lengths[index] = prompt_len;
    return Status::success();
}

[[nodiscard]] compute::BufferView<float> ContiguousKvCache::k_layer(uint32_t layer) const noexcept {
    return _k.view(static_cast<size_t>(layer) * static_cast<size_t>(_layer_stride),
                   static_cast<size_t>(_layer_stride));
}
[[nodiscard]] compute::BufferView<float> ContiguousKvCache::v_layer(uint32_t layer) const noexcept {
    return _v.view(static_cast<size_t>(layer) * static_cast<size_t>(_layer_stride),
                   static_cast<size_t>(_layer_stride));
}
[[nodiscard]] compute::BufferView<float> ContiguousKvCache::k_row(uint32_t layer, uint32_t token) const noexcept {
    const size_t row = static_cast<size_t>(_config.num_kv_heads) * _config.head_dim;
    return _k.view(static_cast<size_t>(layer) * static_cast<size_t>(_layer_stride) +
                       static_cast<size_t>(token) * row,
                   row);
}
[[nodiscard]] compute::BufferView<float> ContiguousKvCache::v_row(uint32_t layer, uint32_t token) const noexcept {
    const size_t row = static_cast<size_t>(_config.num_kv_heads) * _config.head_dim;
    return _v.view(static_cast<size_t>(layer) * static_cast<size_t>(_layer_stride) +
                       static_cast<size_t>(token) * row,
                   row);
}

[[nodiscard]] Status ContiguousKvCache::append_tokens(uint32_t index, uint32_t count) {
    if (index >= _lengths.size()) {
        return Status::error(StatusCode::invalid_argument, "KV sequence index out of range", "index");
    }
    const uint64_t next = static_cast<uint64_t>(_lengths[index]) + count;
    if (next > _config.max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("KV append of {} tokens overruns capacity {} (have {})",
                                           count, _config.max_seq_len, _lengths[index]),
                             "count");
    }
    _lengths[index] = static_cast<uint32_t>(next);
    return Status::success();
}

[[nodiscard]] Status ContiguousKvCache::download_layer(compute::Stream &stream, uint32_t layer, uint32_t index,
                                                       luisa::vector<float> &k_out, luisa::vector<float> &v_out) const {
    if (!valid() || layer >= _config.num_layers || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "KV download index out of range", "layer/index");
    }
    const uint32_t len = _lengths[index];
    const size_t row = static_cast<size_t>(_config.num_kv_heads) * _config.head_dim;
    const size_t count = static_cast<size_t>(len) * row;
    k_out.resize(count);
    v_out.resize(count);
    if (count == 0u) { return Status::success(); }
    // Batch 1: the committed prefix is contiguous from the layer base.
    const size_t base = static_cast<size_t>(layer) * static_cast<size_t>(_layer_stride) +
                        static_cast<size_t>(index) * static_cast<size_t>(_config.max_seq_len) * row;
    stream << _k.view(base, count).copy_to(luisa::span<float>{k_out})
           << _v.view(base, count).copy_to(luisa::span<float>{v_out});
    return Status::success();
}

[[nodiscard]] Status ContiguousKvCache::reset(compute::CommandList &list) {
    if (!valid()) {
        return Status::error(StatusCode::invalid_argument, "KV cache is not created", "cache");
    }
    // DecodeState::record copies the snapshot into heap-owned staging retained
    // through completion, so these stack-local zero vectors are safe even
    // though the list is committed/drained after this call returns.
    const luisa::vector<uint32_t> zeros(_config.batch, 0u);
    const luisa::vector<int32_t> pos_zeros(_config.batch, 0);
    const luisa::vector<uint32_t> states(_config.batch, static_cast<uint32_t>(DecodeSeqState::empty));
    auto s = _state.record(list, zeros, pos_zeros, zeros, states);
    if (!s.ok()) { return s; }
    std::fill(_lengths.begin(), _lengths.end(), 0u);
    return Status::success();
}

[[nodiscard]] Status ContiguousKvCache::record_finished(compute::CommandList &list, uint32_t index) {
    if (!valid() || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "KV sequence index out of range", "index");
    }
    // Device-side completion: kv_len=0, position=0, state=finished. A released
    // slot can then never be attended on reuse (decode attention reads kv_len).
    auto s = _state.record_sequence(list, index, 0u, 0, DecodeSeqState::finished);
    if (!s.ok()) { return s; }
    if (index < _lengths.size()) { _lengths[index] = 0u; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PagedKvCache (optional; fails closed in Phase 3)
// ---------------------------------------------------------------------------

[[nodiscard]] Status PagedKvCache::create(compute::Device & /*device*/, const KvCacheConfig &config,
                                          uint32_t /*page_size*/, MemoryBudget * /*budget*/) {
    _config = config;
    return Status::error(StatusCode::unsupported,
                         "paged KV cache is not implemented in Phase 3; use ContiguousKvCache "
                         "(the contiguous path is the correctness baseline, not a paged fallback)",
                         "paged_kv");
}

}// namespace luisa::ocr