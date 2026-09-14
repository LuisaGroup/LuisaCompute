#pragma once

#include <algorithm>
#include <bit>

#include <luisa/coro/schedulers/wavefront_extension.h>

namespace luisa::compute::coro::detail {

struct WavefrontCoroResumeBatchEntry {
    const WavefrontCoroExtensionStage *stage{nullptr};
    const WavefrontCoroSchedulerExtensionHandler *handler{nullptr};
};

// Runtime batching does not merge compiler boundaries or source spill plans.
// It substitutes a representative handler only for an explicitly permitted,
// physically identical, ordered read-only suffix. Logical value numbers and
// binding owner indices are local to each boundary; compare their resolved
// projections instead. Keep this predicate shared with the negative controls.
[[nodiscard]] inline bool wavefront_coro_resume_batch_compatible(
    luisa::span<const WavefrontCoroResumeBatchEntry> lhs,
    luisa::span<const WavefrontCoroResumeBatchEntry> rhs) noexcept {
    if (lhs.empty() || lhs.size() != rhs.size()) { return false; }
    auto equal = [](auto a, auto b) noexcept {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    };
    size_t target = 0u;
    for (auto i = 0u; i < lhs.size(); ++i) {
        auto [a, ah] = lhs[i];
        auto [b, bh] = rhs[i];
        if (a == nullptr || b == nullptr || ah == nullptr || bh == nullptr ||
            ah->execution() != WavefrontCoroExtensionExecution::before_resume ||
            bh->execution() != WavefrontCoroExtensionExecution::before_resume ||
            ah->batching_identity().empty() ||
            ah->batching_identity() != bh->batching_identity() ||
            a->boundary == nullptr || b->boundary == nullptr ||
            a->extension == nullptr || b->extension == nullptr ||
            a->dataflow == nullptr || b->dataflow == nullptr) { return false; }
        if (a->boundary->to_index == 0u ||
            a->boundary->to_index != b->boundary->to_index ||
            !equal(a->boundary->target_live.slot_span(),
                   b->boundary->target_live.slot_span())) { return false; }
        if (i == 0u) { target = a->boundary->to_index; }
        if (a->boundary->to_index != target) { return false; }
        auto &ae = *a->extension;
        auto &be = *b->extension;
        if (!ae.is_annotation() || !be.is_annotation() ||
            ae.schema() != be.schema() || ae.version() != be.version() ||
            ae.fallback() != be.fallback()) { return false; }
        auto aa = ae.attributes();
        auto ba = be.attributes();
        if (aa.size() != ba.size()) { return false; }
        for (auto j = 0u; j < aa.size(); ++j) {
            if (aa[j].name != ba[j].name ||
                aa[j].value.index() != ba[j].value.index()) { return false; }
            if (auto *av = luisa::get_if<double>(&aa[j].value)) {
                if (std::bit_cast<uint64_t>(*av) !=
                    std::bit_cast<uint64_t>(luisa::get<double>(ba[j].value))) { return false; }
            } else if (aa[j].value != ba[j].value) { return false; }
        }
        auto ab = ae.bindings();
        auto bb = be.bindings();
        if (ab.size() != bb.size()) { return false; }
        for (auto j = 0u; j < ab.size(); ++j) {
            if (ab[j].name != bb[j].name ||
                ab[j].access != CoroSuspendBindingAccess::read ||
                bb[j].access != CoroSuspendBindingAccess::read ||
                ab[j].lifetime != bb[j].lifetime ||
                ab[j].index >= a->boundary->bindings.size() ||
                bb[j].index >= b->boundary->bindings.size()) { return false; }
            auto &ap = a->boundary->bindings[ab[j].index];
            auto &bp = b->boundary->bindings[bb[j].index];
            if (ap.type() != bp.type() ||
                ap.access() != CoroSuspendBindingAccess::read ||
                bp.access() != CoroSuspendBindingAccess::read ||
                ap.lifetime() != bp.lifetime() ||
                !equal(ap.use_slots(), bp.use_slots()) ||
                !equal(ap.def_slots(), bp.def_slots()) ||
                !equal(ap.rmw_slots(), bp.rmw_slots()) ||
                !equal(ap.reconstruct_slots(), bp.reconstruct_slots())) { return false; }
            auto ax = ap.pieces();
            auto bx = bp.pieces();
            // An empty flat projection is not an equivalence certificate.
            // In particular, shared-callable alias bindings may be resolved
            // by private guarded alternatives rather than pieces(). Keep
            // them separate until their complete condition trees are exposed
            // and compared, even if their union of resident slots is equal.
            if (ax.empty() || bx.empty() || ax.size() != bx.size()) { return false; }
            for (auto k = 0u; k < ax.size(); ++k) {
                if (ax[k].field_index != bx[k].field_index ||
                    ax[k].logical_type != bx[k].logical_type ||
                    ax[k].physical_type != bx[k].physical_type ||
                    ax[k].bit_offset != bx[k].bit_offset ||
                    !equal(luisa::span{ax[k].access_chain},
                           luisa::span{bx[k].access_chain})) { return false; }
            }
        }
        auto &ad = *a->dataflow;
        auto &bd = *b->dataflow;
        if (!ad.def.slots.empty() || !bd.def.slots.empty() ||
            !equal(ad.use.slot_span(), bd.use.slot_span()) ||
            !equal(ad.live_in.slot_span(), bd.live_in.slot_span()) ||
            !equal(ad.live_out.slot_span(), bd.live_out.slot_span()) ||
            !equal(ad.preserve.slot_span(), bd.preserve.slot_span()) ||
            !equal(ad.required_def.slot_span(), bd.required_def.slot_span()) ||
            !equal(ad.rmw_slot_span(), bd.rmw_slot_span()) ||
            !equal(ad.reconstruct_slot_span(), bd.reconstruct_slot_span())) { return false; }
    }
    return true;
}

}// namespace luisa::compute::coro::detail
