#include "coro_guarded_scalar_relation.h"

#include <algorithm>
#include <limits>

#include <luisa/core/stl/hash.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/value.h>

namespace luisa::compute::xir::detail {

namespace {

constexpr size_t max_bdd_nodes = 1u << 20u;
constexpr auto terminal_variable = std::numeric_limits<uint32_t>::max();

[[nodiscard]] uint64_t pair_key(uint32_t lhs, uint32_t rhs) noexcept {
    if (rhs < lhs) { std::swap(lhs, rhs); }
    return static_cast<uint64_t>(lhs) |
           (static_cast<uint64_t>(rhs) << 32u);
}

}// namespace

CoroBooleanSetManager::CoroBooleanSetManager() noexcept {
    _nodes.emplace_back(Node{terminal_variable, 0u, 0u});
    _nodes.emplace_back(Node{terminal_variable, 1u, 1u});
}

uint32_t CoroBooleanSetManager::_variable(
    Value *predicate) noexcept {
    if (auto iter = _variables.find(predicate);
        iter != _variables.end()) {
        return iter->second;
    }
    auto id = static_cast<uint32_t>(_variables.size());
    _variables.emplace(predicate, id);
    _variable_values.emplace_back(predicate);
    return id;
}

CoroBooleanSetManager::Set CoroBooleanSetManager::_make_node(
    uint32_t variable, Set low, Set high) noexcept {
    if (low == high) { return low; }
    if (_nodes.size() >= max_bdd_nodes) {
        _widened = true;
        return universe();
    }
    uint64_t words[]{variable, low, high};
    auto hash = luisa::hash64(words, sizeof(words), hash64_default_seed);
    auto &bucket = _node_buckets[hash];
    for (auto candidate : bucket) {
        auto node = _nodes[candidate];
        if (node.variable == variable &&
            node.low == low && node.high == high) {
            return candidate;
        }
    }
    auto id = static_cast<Set>(_nodes.size());
    _nodes.emplace_back(Node{variable, low, high});
    bucket.emplace_back(id);
    return id;
}

CoroBooleanSetManager::Set CoroBooleanSetManager::_apply(
    bool is_union, Set lhs, Set rhs) noexcept {
    if (rhs < lhs) { std::swap(lhs, rhs); }
    if (is_union) {
        if (lhs == empty_set() || lhs == rhs) { return rhs; }
        if (rhs == universe()) { return universe(); }
    } else {
        if (lhs == empty_set()) { return empty_set(); }
        if (lhs == rhs || rhs == universe()) { return lhs; }
    }
    auto key = pair_key(lhs, rhs);
    auto &cache = is_union ? _union_cache : _intersection_cache;
    if (auto iter = cache.find(key); iter != cache.end()) {
        return iter->second;
    }
    auto lhs_node = _nodes[lhs];
    auto rhs_node = _nodes[rhs];
    auto variable = std::min(lhs_node.variable, rhs_node.variable);
    auto lhs_low = lhs_node.variable == variable ? lhs_node.low : lhs;
    auto lhs_high = lhs_node.variable == variable ? lhs_node.high : lhs;
    auto rhs_low = rhs_node.variable == variable ? rhs_node.low : rhs;
    auto rhs_high = rhs_node.variable == variable ? rhs_node.high : rhs;
    auto low = _apply(is_union, lhs_low, rhs_low);
    auto high = _apply(is_union, lhs_high, rhs_high);
    auto result = _make_node(variable, low, high);
    cache.emplace(key, result);
    return result;
}

CoroBooleanSetManager::Set CoroBooleanSetManager::_forget(
    Set set, uint32_t variable) noexcept {
    luisa::unordered_map<Set, Set> cache;
    const auto visit = [&](auto &&self, Set current) noexcept -> Set {
        if (current <= universe()) { return current; }
        if (auto iter = cache.find(current); iter != cache.end()) {
            return iter->second;
        }
        auto node = _nodes[current];
        Set result;
        if (node.variable > variable) {
            result = current;
        } else if (node.variable == variable) {
            result = unite(node.low, node.high);
        } else {
            result = _make_node(
                node.variable,
                self(self, node.low),
                self(self, node.high));
        }
        cache.emplace(current, result);
        return result;
    };
    return visit(visit, set);
}

CoroBooleanSetManager::Set CoroBooleanSetManager::literal(
    Value *predicate, bool value) noexcept {
    if (predicate == nullptr) { return universe(); }
    auto variable = _variable(predicate);
    return value ? _make_node(variable, empty_set(), universe()) :
                   _make_node(variable, universe(), empty_set());
}

CoroBooleanSetManager::Set CoroBooleanSetManager::unite(
    Set lhs, Set rhs) noexcept {
    return _apply(true, lhs, rhs);
}

CoroBooleanSetManager::Set CoroBooleanSetManager::intersect(
    Set lhs, Set rhs) noexcept {
    return _apply(false, lhs, rhs);
}

CoroBooleanSetManager::Set CoroBooleanSetManager::forget(
    Set set, Value *predicate) noexcept {
    if (predicate == nullptr) { return set; }
    auto iter = _variables.find(predicate);
    return iter == _variables.end() ? set :
                                      _forget(set, iter->second);
}

CoroBooleanSetManager::Set CoroBooleanSetManager::assign(
    Set set, Value *destination, Value *source,
    bool true_when_source_is,
    luisa::optional<bool> copied_constant) noexcept {
    if (destination == nullptr) { return set; }
    if (source == destination && !copied_constant) {
        return true_when_source_is ? set : forget(set, destination);
    }
    auto result = forget(set, destination);
    if (copied_constant) {
        return intersect(result, literal(destination, *copied_constant));
    }
    if (source == nullptr) { return result; }
    auto destination_true = literal(destination, true);
    auto destination_false = literal(destination, false);
    auto source_true = literal(source, true_when_source_is);
    auto source_false = literal(source, !true_when_source_is);
    auto equality = unite(
        intersect(destination_true, source_true),
        intersect(destination_false, source_false));
    return intersect(result, equality);
}

luisa::string CoroBooleanSetManager::describe(Set set) const noexcept {
    auto emitted = size_t{0u};
    const auto visit = [&](auto &&self, Set current) noexcept -> luisa::string {
        if (current == empty_set()) { return "false"; }
        if (current == universe()) { return "true"; }
        if (++emitted > 128u) { return "..."; }
        auto node = _nodes[current];
        auto *value = node.variable < _variable_values.size() ?
                          _variable_values[node.variable] :
                          nullptr;
        auto name = value == nullptr ?
                        luisa::string{"<unknown>"} :
                        luisa::string{value->name().value_or("<unnamed>")};
        return "(" + name + " ? " + self(self, node.high) +
               " : " + self(self, node.low) + ")";
    };
    return visit(visit, set);
}

CoroGuardedScalarRelationDomain::CoroGuardedScalarRelationDomain(
    CoroBooleanSetManager &manager,
    luisa::span<const CoroMaskedScalarWitness>
        masked_scalar_witnesses,
    luisa::span<const CoroCounterAvailabilityWitness>
        counter_availability_witnesses) noexcept
    : _manager{&manager} {
    for (auto witness : masked_scalar_witnesses) {
        if (witness.scalar != nullptr && witness.mask != 0u) {
            _masked_nonzero_possible[witness.scalar]
                                       .try_emplace(
                                           witness.mask,
                                           _feasible);
            _masked_nonzero_positive_unsafe[witness.scalar]
                                              .try_emplace(
                                                  witness.mask,
                                                  _feasible);
        }
    }
    for (auto witness : counter_availability_witnesses) {
        if (witness.scalar != nullptr) {
            _scalar_zero_possible.try_emplace(
                witness.scalar, _feasible);
            _scalar_zero_positive_unsafe.try_emplace(
                witness.scalar, _feasible);
        }
    }
}

CoroGuardedScalarRelationDomain::Set
CoroGuardedScalarRelationDomain::_unsafe(
    const luisa::unordered_map<AllocaInst *, Set> &relations,
    AllocaInst *index) const noexcept {
    if (index != nullptr) {
        if (auto iter = relations.find(index); iter != relations.end()) {
            return iter->second;
        }
    }
    return _feasible;
}

void CoroGuardedScalarRelationDomain::_canonicalize(
    luisa::unordered_map<AllocaInst *, Set> &relations) noexcept {
    for (auto iter = relations.begin(); iter != relations.end();) {
        if (iter->second == _feasible) {
            iter = relations.erase(iter);
        } else {
            ++iter;
        }
    }
}

luisa::unordered_map<uint64_t, CoroGuardedScalarRelationDomain::Set>
CoroGuardedScalarRelationDomain::_masked_possibilities(
    AllocaInst *scalar) const noexcept {
    if (auto iter = _masked_nonzero_possible.find(scalar);
        iter != _masked_nonzero_possible.end()) {
        return iter->second;
    }
    return {};
}

luisa::unordered_map<uint64_t, CoroGuardedScalarRelationDomain::Set>
CoroGuardedScalarRelationDomain::_masked_witnesses(
    AllocaInst *scalar) const noexcept {
    if (auto iter = _masked_nonzero_positive_unsafe.find(scalar);
        iter != _masked_nonzero_positive_unsafe.end()) {
        return iter->second;
    }
    return {};
}

bool CoroGuardedScalarRelationDomain::knows_less(
    AllocaInst *index) const noexcept {
    return CoroBooleanSetManager::is_empty(
        _unsafe(_less_unsafe, index));
}

bool CoroGuardedScalarRelationDomain::knows_equal(
    AllocaInst *index) const noexcept {
    return CoroBooleanSetManager::is_empty(
        _unsafe(_equal_unsafe, index));
}

bool CoroGuardedScalarRelationDomain::knows_last(
    AllocaInst *index) const noexcept {
    return CoroBooleanSetManager::is_empty(
        _unsafe(_last_unsafe, index));
}

bool CoroGuardedScalarRelationDomain::knows_counter_positive() const noexcept {
    return CoroBooleanSetManager::is_empty(
        _counter_positive_unsafe);
}

bool CoroGuardedScalarRelationDomain::knows_tail() const noexcept {
    return CoroBooleanSetManager::is_empty(_tail_unsafe);
}

bool CoroGuardedScalarRelationDomain::knows_initialized(
    AllocaInst *index) const noexcept {
    // Safe(I) = (I<C) || (I=C && Tail(A,C)). Therefore the unsafe
    // valuations are !Less && (!Equal || !Tail).
    auto derived_unsafe = _manager->intersect(
        _unsafe(_less_unsafe, index),
        _manager->unite(
            _unsafe(_equal_unsafe, index), _tail_unsafe));
    auto unsafe = _manager->intersect(
        _unsafe(_initialized_unsafe, index), derived_unsafe);
    return CoroBooleanSetManager::is_empty(unsafe);
}

bool CoroGuardedScalarRelationDomain::tracks_masked_scalar(
    AllocaInst *scalar) const noexcept {
    return scalar != nullptr &&
           _masked_nonzero_positive_unsafe.contains(scalar);
}

bool CoroGuardedScalarRelationDomain::
masked_nonzero_implies_counter_positive(
    AllocaInst *scalar, uint64_t mask) const noexcept {
    if (auto scalar_iter =
            _masked_nonzero_positive_unsafe.find(scalar);
        scalar_iter != _masked_nonzero_positive_unsafe.end()) {
        if (auto mask_iter = scalar_iter->second.find(mask);
            mask_iter != scalar_iter->second.end()) {
            return CoroBooleanSetManager::is_empty(mask_iter->second);
        }
    }
    return false;
}

bool CoroGuardedScalarRelationDomain::refine_masked_scalar_nonzero(
    AllocaInst *scalar, uint64_t mask) noexcept {
    // N(S, M) over-approximates the Boolean valuations on which S&M may be
    // nonzero. A concrete state selected by S&M!=0 is therefore contained in
    // F intersect N. Restricting the entire product domain to N is sound even
    // when N is imprecise; an empty intersection proves the edge
    // unreachable. A missing witness conservatively denotes N=F.
    auto selected = _feasible;
    if (auto scalar_iter = _masked_nonzero_possible.find(scalar);
        scalar_iter != _masked_nonzero_possible.end()) {
        if (auto mask_iter = scalar_iter->second.find(mask);
            mask_iter != scalar_iter->second.end()) {
            selected = mask_iter->second;
        }
    }
    return _refine(selected);
}

bool CoroGuardedScalarRelationDomain::tracks_counter_availability(
    AllocaInst *scalar) const noexcept {
    return scalar != nullptr &&
           _scalar_zero_positive_unsafe.contains(scalar);
}

bool CoroGuardedScalarRelationDomain::
scalar_zero_implies_counter_positive(
    AllocaInst *scalar) const noexcept {
    if (auto iter = _scalar_zero_positive_unsafe.find(scalar);
        iter != _scalar_zero_positive_unsafe.end()) {
        return CoroBooleanSetManager::is_empty(iter->second);
    }
    return false;
}

bool CoroGuardedScalarRelationDomain::refine_scalar_zero(
    AllocaInst *scalar) noexcept {
    // Z(S) over-approximates the valuations on which S may be zero. The
    // selected S==0 edge is contained in F intersect Z(S); restricting the
    // full product state to that set is therefore sound, and an empty result
    // proves the edge unreachable.
    auto selected = _feasible;
    if (auto iter = _scalar_zero_possible.find(scalar);
        iter != _scalar_zero_possible.end()) {
        selected = iter->second;
    }
    return _refine(selected);
}

void CoroGuardedScalarRelationDomain::assume_scalar_nonzero(
    AllocaInst *scalar) noexcept {
    if (auto possible = _scalar_zero_possible.find(scalar);
        possible != _scalar_zero_possible.end()) {
        possible->second = CoroBooleanSetManager::empty_set();
        _scalar_zero_positive_unsafe[scalar] =
            CoroBooleanSetManager::empty_set();
    }
}

void CoroGuardedScalarRelationDomain::materialize_initialized() noexcept {
    for (auto *index : _tracked_indices) {
        auto derived_unsafe = _manager->intersect(
            _unsafe(_less_unsafe, index),
            _manager->unite(
                _unsafe(_equal_unsafe, index), _tail_unsafe));
        auto unsafe = _manager->intersect(
            _unsafe(_initialized_unsafe, index), derived_unsafe);
        if (unsafe == _feasible) {
            _initialized_unsafe.erase(index);
        } else {
            _initialized_unsafe[index] = unsafe;
        }
    }
}

void CoroGuardedScalarRelationDomain::add_less(
    AllocaInst *index) noexcept {
    if (index != nullptr) {
        _tracked_indices.emplace(index);
        _less_unsafe[index] = CoroBooleanSetManager::empty_set();
    }
}

void CoroGuardedScalarRelationDomain::add_equal(
    AllocaInst *index) noexcept {
    if (index != nullptr) {
        _tracked_indices.emplace(index);
        _equal_unsafe[index] = CoroBooleanSetManager::empty_set();
    }
}

void CoroGuardedScalarRelationDomain::add_last(
    AllocaInst *index) noexcept {
    if (index != nullptr) {
        _tracked_indices.emplace(index);
        _last_unsafe[index] = CoroBooleanSetManager::empty_set();
    }
}

void CoroGuardedScalarRelationDomain::erase_index(
    AllocaInst *index) noexcept {
    if (index == nullptr) { return; }
    _less_unsafe.erase(index);
    _equal_unsafe.erase(index);
    _last_unsafe.erase(index);
    _initialized_unsafe.erase(index);
    _tracked_indices.erase(index);
}

void CoroGuardedScalarRelationDomain::retain_indices(
    luisa::span<AllocaInst *const> live_indices) noexcept {
    const auto is_live = [live_indices](AllocaInst *index) noexcept {
        return std::find(live_indices.begin(), live_indices.end(), index) !=
               live_indices.end();
    };
    for (auto iter = _tracked_indices.begin();
         iter != _tracked_indices.end();) {
        if (!is_live(*iter)) {
            _less_unsafe.erase(*iter);
            _equal_unsafe.erase(*iter);
            _last_unsafe.erase(*iter);
            _initialized_unsafe.erase(*iter);
            iter = _tracked_indices.erase(iter);
        } else {
            ++iter;
        }
    }
}

void CoroGuardedScalarRelationDomain::clear_relations() noexcept {
    _less_unsafe.clear();
    _equal_unsafe.clear();
    _last_unsafe.clear();
    // The current scalar values remain tracked after a counter reset; their
    // counter relations simply become unknown under the current feasible
    // set. Materialized physical initialization is intentionally preserved.
}

void CoroGuardedScalarRelationDomain::add_counter_positive() noexcept {
    _counter_positive_unsafe = CoroBooleanSetManager::empty_set();
    // Once C>0 holds throughout the current feasible state, every
    // implication (S & M)!=0 => C>0 is true independently of S.
    for (auto &[_, masks] : _masked_nonzero_positive_unsafe) {
        for (auto &[__, unsafe] : masks) {
            unsafe = CoroBooleanSetManager::empty_set();
        }
    }
    for (auto &[_, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = CoroBooleanSetManager::empty_set();
    }
}

void CoroGuardedScalarRelationDomain::clear_counter_positive() noexcept {
    _counter_positive_unsafe = _feasible;
    // Losing C>0 does not make every masked implication unknown. For each
    // witness, violation is possible exactly where S&M may be nonzero. This
    // distinction is essential when S&M==0 was established independently of
    // the counter (for example by an ABI precondition before C:=0).
    for (auto &[scalar, masks] :
         _masked_nonzero_positive_unsafe) {
        auto possibilities = _masked_possibilities(scalar);
        for (auto &[mask, unsafe] : masks) {
            unsafe = possibilities.contains(mask) ?
                         possibilities.find(mask)->second :
                         _feasible;
        }
    }
    for (auto &[scalar, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = _scalar_zero_possible.contains(scalar) ?
                     _scalar_zero_possible.find(scalar)->second :
                     _feasible;
    }
}

void CoroGuardedScalarRelationDomain::add_tail() noexcept {
    _tail_unsafe = CoroBooleanSetManager::empty_set();
}

void CoroGuardedScalarRelationDomain::clear_tail() noexcept {
    _tail_unsafe = _feasible;
}

void CoroGuardedScalarRelationDomain::advance_counter() noexcept {
    luisa::vector<AllocaInst *> indices;
    indices.reserve(_less_unsafe.size() + _equal_unsafe.size());
    for (auto [index, _] : _less_unsafe) { indices.emplace_back(index); }
    for (auto [index, _] : _equal_unsafe) {
        if (std::find(indices.begin(), indices.end(), index) ==
            indices.end()) {
            indices.emplace_back(index);
        }
    }
    for (auto *index : indices) {
        auto unsafe = _manager->intersect(
            _unsafe(_less_unsafe, index),
            _unsafe(_equal_unsafe, index));
        if (unsafe == _feasible) {
            _less_unsafe.erase(index);
        } else {
            _less_unsafe[index] = unsafe;
        }
    }
    // I == old(C) is exactly I + 1 == new(C). Preserve that stronger
    // identity separately from the less-than consequence used by ordinary
    // prefix reads.
    _last_unsafe = _equal_unsafe;
    _equal_unsafe.clear();
    // A separately proved non-wrapping unsigned increment always produces a
    // value of at least one, independently of the old counter value.
    add_counter_positive();
}

void CoroGuardedScalarRelationDomain::retreat_counter() noexcept {
    // This operation is called only after C>0 has been proved, so
    // new(C) = old(C) - 1 cannot wrap. Hence
    //
    //   I + 1 == old(C)  <=>  I == new(C).
    //
    // A generic I < old(C) yields only I <= new(C), which is deliberately
    // not strengthened to either equality or strict inequality.
    _equal_unsafe = _last_unsafe;
    _less_unsafe.clear();
    _last_unsafe.clear();
    // C>0 before the decrement proves only new(C)>=0. If new(C) can be zero,
    // a previously nonzero witness scalar no longer implies positivity.
    invalidate_counter_implications();
}

void CoroGuardedScalarRelationDomain::
invalidate_counter_implications() noexcept {
    for (auto &[scalar, masks] : _masked_nonzero_positive_unsafe) {
        auto possibilities = _masked_possibilities(scalar);
        for (auto &[mask, unsafe] : masks) {
            unsafe = possibilities.contains(mask) ?
                         possibilities.find(mask)->second :
                         _feasible;
        }
    }
    for (auto &[scalar, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = _scalar_zero_possible.contains(scalar) ?
                     _scalar_zero_possible.find(scalar)->second :
                     _feasible;
    }
}

void CoroGuardedScalarRelationDomain::assume_masked_scalar_zero(
    AllocaInst *scalar, uint64_t zero_mask) noexcept {
    if (auto iter = _masked_nonzero_positive_unsafe.find(scalar);
        iter != _masked_nonzero_positive_unsafe.end()) {
        auto &possibilities = _masked_nonzero_possible.at(scalar);
        for (auto &[mask, unsafe] : iter->second) {
            if ((mask & ~zero_mask) == 0u) {
                // S&M is exactly zero, so the implication is vacuous.
                possibilities[mask] =
                    CoroBooleanSetManager::empty_set();
                unsafe = CoroBooleanSetManager::empty_set();
            }
        }
    }
}

luisa::vector<uint64_t>
CoroGuardedScalarRelationDomain::masked_scalar_masks(
    AllocaInst *scalar) const noexcept {
    luisa::vector<uint64_t> result;
    if (auto iter = _masked_nonzero_positive_unsafe.find(scalar);
        iter != _masked_nonzero_positive_unsafe.end()) {
        result.reserve(iter->second.size());
        for (auto [mask, _] : iter->second) {
            result.emplace_back(mask);
        }
    }
    return result;
}

CoroMaskedScalarProjection
CoroGuardedScalarRelationDomain::masked_scalar_unknown_projection()
    const noexcept {
    return CoroMaskedScalarProjection{
        .nonzero_possible = _feasible,
        .nonzero_without_positive_counter = _counter_positive_unsafe};
}

CoroMaskedScalarProjection
CoroGuardedScalarRelationDomain::masked_scalar_constant_projection(
    uint64_t value, uint64_t mask) const noexcept {
    if ((value & mask) == 0u) {
        return CoroMaskedScalarProjection{
            .nonzero_possible = CoroBooleanSetManager::empty_set(),
            .nonzero_without_positive_counter =
                CoroBooleanSetManager::empty_set()};
    }
    return masked_scalar_unknown_projection();
}

CoroMaskedScalarProjection
CoroGuardedScalarRelationDomain::masked_scalar_load_projection(
    AllocaInst *source, uint64_t mask) const noexcept {
    auto result = masked_scalar_unknown_projection();
    if (auto scalar_iter = _masked_nonzero_possible.find(source);
        scalar_iter != _masked_nonzero_possible.end()) {
        if (auto mask_iter = scalar_iter->second.find(mask);
            mask_iter != scalar_iter->second.end()) {
            result.nonzero_possible = mask_iter->second;
        }
    }
    if (auto scalar_iter =
            _masked_nonzero_positive_unsafe.find(source);
        scalar_iter != _masked_nonzero_positive_unsafe.end()) {
        if (auto mask_iter = scalar_iter->second.find(mask);
            mask_iter != scalar_iter->second.end()) {
            result.nonzero_without_positive_counter = mask_iter->second;
        }
    }
    return result;
}

CoroMaskedScalarProjection
CoroGuardedScalarRelationDomain::masked_scalar_projection_union(
    CoroMaskedScalarProjection lhs,
    CoroMaskedScalarProjection rhs) const noexcept {
    // OR, XOR, and an uncorrelated SELECT can be nonzero because either
    // operand/arm is nonzero. The union of their may-sets therefore
    // over-approximates both result nonzero and result nonzero with C==0.
    return CoroMaskedScalarProjection{
        .nonzero_possible = _manager->unite(
            lhs.nonzero_possible, rhs.nonzero_possible),
        .nonzero_without_positive_counter = _manager->unite(
            lhs.nonzero_without_positive_counter,
            rhs.nonzero_without_positive_counter)};
}

CoroMaskedScalarProjection
CoroGuardedScalarRelationDomain::masked_scalar_projection_intersection(
    CoroMaskedScalarProjection lhs,
    CoroMaskedScalarProjection rhs) const noexcept {
    // (x & y) & M != 0 implies both x&M != 0 and y&M != 0. Likewise, a
    // violating state additionally has the same C==0 in both operand
    // projections. Intersecting their respective over-approximations is
    // therefore sound and strictly more precise than picking one operand.
    return CoroMaskedScalarProjection{
        .nonzero_possible = _manager->intersect(
            lhs.nonzero_possible, rhs.nonzero_possible),
        .nonzero_without_positive_counter = _manager->intersect(
            lhs.nonzero_without_positive_counter,
            rhs.nonzero_without_positive_counter)};
}

void CoroGuardedScalarRelationDomain::assign_masked_scalar_projection(
    AllocaInst *destination, uint64_t mask,
    CoroMaskedScalarProjection projection) noexcept {
    auto possible_iter = _masked_nonzero_possible.find(destination);
    auto unsafe_iter =
        _masked_nonzero_positive_unsafe.find(destination);
    if (possible_iter == _masked_nonzero_possible.end() ||
        unsafe_iter == _masked_nonzero_positive_unsafe.end() ||
        !possible_iter->second.contains(mask) ||
        !unsafe_iter->second.contains(mask)) {
        return;
    }
    possible_iter->second[mask] = projection.nonzero_possible;
    unsafe_iter->second[mask] =
        projection.nonzero_without_positive_counter;
}

CoroScalarZeroProjection
CoroGuardedScalarRelationDomain::scalar_zero_unknown_projection()
    const noexcept {
    return CoroScalarZeroProjection{
        .zero_possible = _feasible,
        .zero_without_positive_counter = _counter_positive_unsafe};
}

CoroScalarZeroProjection
CoroGuardedScalarRelationDomain::scalar_zero_constant_projection(
    uint64_t value) const noexcept {
    if (value != 0u) {
        return CoroScalarZeroProjection{
            .zero_possible = CoroBooleanSetManager::empty_set(),
            .zero_without_positive_counter =
                CoroBooleanSetManager::empty_set()};
    }
    return scalar_zero_unknown_projection();
}

CoroScalarZeroProjection
CoroGuardedScalarRelationDomain::scalar_zero_load_projection(
    AllocaInst *source) const noexcept {
    auto result = scalar_zero_unknown_projection();
    if (auto iter = _scalar_zero_possible.find(source);
        iter != _scalar_zero_possible.end()) {
        result.zero_possible = iter->second;
    }
    if (auto iter = _scalar_zero_positive_unsafe.find(source);
        iter != _scalar_zero_positive_unsafe.end()) {
        result.zero_without_positive_counter = iter->second;
    }
    return result;
}

CoroScalarZeroProjection
CoroGuardedScalarRelationDomain::scalar_zero_projection_union(
    CoroScalarZeroProjection lhs,
    CoroScalarZeroProjection rhs) const noexcept {
    return CoroScalarZeroProjection{
        .zero_possible = _manager->unite(
            lhs.zero_possible, rhs.zero_possible),
        .zero_without_positive_counter = _manager->unite(
            lhs.zero_without_positive_counter,
            rhs.zero_without_positive_counter)};
}

CoroScalarZeroProjection
CoroGuardedScalarRelationDomain::scalar_zero_projection_intersection(
    CoroScalarZeroProjection lhs,
    CoroScalarZeroProjection rhs) const noexcept {
    return CoroScalarZeroProjection{
        .zero_possible = _manager->intersect(
            lhs.zero_possible, rhs.zero_possible),
        .zero_without_positive_counter = _manager->intersect(
            lhs.zero_without_positive_counter,
            rhs.zero_without_positive_counter)};
}

void CoroGuardedScalarRelationDomain::assign_scalar_zero_projection(
    AllocaInst *destination,
    CoroScalarZeroProjection projection) noexcept {
    if (!_scalar_zero_possible.contains(destination) ||
        !_scalar_zero_positive_unsafe.contains(destination)) {
        return;
    }
    _scalar_zero_possible[destination] = projection.zero_possible;
    _scalar_zero_positive_unsafe[destination] =
        projection.zero_without_positive_counter;
}

void CoroGuardedScalarRelationDomain::forget_boolean(
    Value *predicate) noexcept {
    assign_boolean(predicate, nullptr, true, luisa::nullopt);
}

void CoroGuardedScalarRelationDomain::retain_booleans(
    luisa::span<Value *const> live_predicates) noexcept {
    const auto is_live = [live_predicates](Value *predicate) noexcept {
        return std::find(live_predicates.begin(), live_predicates.end(),
                         predicate) != live_predicates.end();
    };
    luisa::vector<Value *> dead;
    dead.reserve(_tracked_predicates.size());
    for (auto *predicate : _tracked_predicates) {
        if (!is_live(predicate)) { dead.emplace_back(predicate); }
    }
    if (dead.empty()) { return; }
    luisa::unordered_map<Set, Set> projected;
    const auto project = [&](Set set) noexcept {
        if (auto iter = projected.find(set); iter != projected.end()) {
            return iter->second;
        }
        auto result = set;
        for (auto *predicate : dead) {
            result = _manager->forget(result, predicate);
        }
        projected.emplace(set, result);
        return result;
    };
    _feasible = project(_feasible);
    _counter_positive_unsafe = project(_counter_positive_unsafe);
    _tail_unsafe = project(_tail_unsafe);
    for (auto &[_, unsafe] : _less_unsafe) { unsafe = project(unsafe); }
    for (auto &[_, unsafe] : _equal_unsafe) { unsafe = project(unsafe); }
    for (auto &[_, unsafe] : _last_unsafe) { unsafe = project(unsafe); }
    for (auto &[_, unsafe] : _initialized_unsafe) {
        unsafe = project(unsafe);
    }
    for (auto &[_, masks] : _masked_nonzero_possible) {
        for (auto &[__, possible] : masks) {
            possible = project(possible);
        }
    }
    for (auto &[_, masks] : _masked_nonzero_positive_unsafe) {
        for (auto &[__, unsafe] : masks) {
            unsafe = project(unsafe);
        }
    }
    for (auto &[_, possible] : _scalar_zero_possible) {
        possible = project(possible);
    }
    for (auto &[_, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = project(unsafe);
    }
    for (auto *predicate : dead) {
        _tracked_predicates.erase(predicate);
    }
    _canonicalize(_less_unsafe);
    _canonicalize(_equal_unsafe);
    _canonicalize(_last_unsafe);
    _canonicalize(_initialized_unsafe);
}

void CoroGuardedScalarRelationDomain::assign_boolean(
    Value *destination, Value *source,
    bool true_when_source_is,
    luisa::optional<bool> copied_constant) noexcept {
    if (destination != nullptr) {
        _tracked_predicates.emplace(destination);
    }
    if (source != nullptr) { _tracked_predicates.emplace(source); }
    luisa::unordered_map<Set, Set> transformed;
    const auto transform = [&](Set set) noexcept {
        if (auto iter = transformed.find(set); iter != transformed.end()) {
            return iter->second;
        }
        auto result = _manager->assign(
            set, destination, source,
            true_when_source_is, copied_constant);
        transformed.emplace(set, result);
        return result;
    };
    _feasible = transform(_feasible);
    _counter_positive_unsafe = transform(_counter_positive_unsafe);
    _tail_unsafe = transform(_tail_unsafe);
    for (auto &[_, unsafe] : _less_unsafe) { unsafe = transform(unsafe); }
    for (auto &[_, unsafe] : _equal_unsafe) { unsafe = transform(unsafe); }
    for (auto &[_, unsafe] : _last_unsafe) { unsafe = transform(unsafe); }
    for (auto &[_, unsafe] : _initialized_unsafe) {
        unsafe = transform(unsafe);
    }
    for (auto &[_, masks] : _masked_nonzero_possible) {
        for (auto &[__, possible] : masks) {
            possible = transform(possible);
        }
    }
    for (auto &[_, masks] : _masked_nonzero_positive_unsafe) {
        for (auto &[__, unsafe] : masks) {
            unsafe = transform(unsafe);
        }
    }
    for (auto &[_, possible] : _scalar_zero_possible) {
        possible = transform(possible);
    }
    for (auto &[_, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = transform(unsafe);
    }
    _canonicalize(_less_unsafe);
    _canonicalize(_equal_unsafe);
    _canonicalize(_last_unsafe);
    _canonicalize(_initialized_unsafe);
}

bool CoroGuardedScalarRelationDomain::_refine(Set selected) noexcept {
    _feasible = _manager->intersect(_feasible, selected);
    if (!feasible()) { return false; }
    _counter_positive_unsafe = _manager->intersect(
        _counter_positive_unsafe, selected);
    _tail_unsafe = _manager->intersect(_tail_unsafe, selected);
    for (auto &[_, unsafe] : _less_unsafe) {
        unsafe = _manager->intersect(unsafe, selected);
    }
    for (auto &[_, unsafe] : _equal_unsafe) {
        unsafe = _manager->intersect(unsafe, selected);
    }
    for (auto &[_, unsafe] : _last_unsafe) {
        unsafe = _manager->intersect(unsafe, selected);
    }
    for (auto &[_, unsafe] : _initialized_unsafe) {
        unsafe = _manager->intersect(unsafe, selected);
    }
    for (auto &[_, masks] : _masked_nonzero_possible) {
        for (auto &[__, possible] : masks) {
            possible = _manager->intersect(possible, selected);
        }
    }
    for (auto &[_, masks] : _masked_nonzero_positive_unsafe) {
        for (auto &[__, unsafe] : masks) {
            unsafe = _manager->intersect(unsafe, selected);
        }
    }
    for (auto &[_, possible] : _scalar_zero_possible) {
        possible = _manager->intersect(possible, selected);
    }
    for (auto &[_, unsafe] : _scalar_zero_positive_unsafe) {
        unsafe = _manager->intersect(unsafe, selected);
    }
    _canonicalize(_less_unsafe);
    _canonicalize(_equal_unsafe);
    _canonicalize(_last_unsafe);
    _canonicalize(_initialized_unsafe);
    return true;
}

bool CoroGuardedScalarRelationDomain::refine_boolean(
    Value *predicate, bool value) noexcept {
    if (predicate != nullptr) {
        _tracked_predicates.emplace(predicate);
    }
    return _refine(_manager->literal(predicate, value));
}

void CoroGuardedScalarRelationDomain::assign_index_copy(
    AllocaInst *destination, AllocaInst *source,
    bool source_is_counter) noexcept {
    if (destination == nullptr || destination == source) { return; }
    auto less = _unsafe(_less_unsafe, source);
    auto equal = source_is_counter ?
                     CoroBooleanSetManager::empty_set() :
                     _unsafe(_equal_unsafe, source);
    auto last = source_is_counter ?
                    _feasible :
                    _unsafe(_last_unsafe, source);
    auto initialized = source_is_counter ?
                           _feasible :
                           _unsafe(_initialized_unsafe, source);
    _less_unsafe.erase(destination);
    _equal_unsafe.erase(destination);
    _last_unsafe.erase(destination);
    _initialized_unsafe.erase(destination);
    _tracked_indices.erase(destination);
    _tracked_indices.emplace(destination);
    if (less != _feasible) { _less_unsafe[destination] = less; }
    if (equal != _feasible) { _equal_unsafe[destination] = equal; }
    if (last != _feasible) { _last_unsafe[destination] = last; }
    if (initialized != _feasible) {
        _initialized_unsafe[destination] = initialized;
    }
}

bool CoroGuardedScalarRelationDomain::merge(
    const CoroGuardedScalarRelationDomain &incoming) noexcept {
    auto before = *this;
    auto old_feasible = _feasible;
    auto merged_feasible = _manager->unite(
        old_feasible, incoming._feasible);
    auto merged_counter_positive_unsafe = _manager->unite(
        _counter_positive_unsafe,
        incoming._counter_positive_unsafe);
    auto merged_tail_unsafe = _manager->unite(
        _tail_unsafe, incoming._tail_unsafe);
    const auto merge_relations = [&](auto &target, auto &&source) noexcept {
        luisa::vector<AllocaInst *> indices;
        indices.reserve(target.size() + source.size());
        for (auto [index, _] : target) { indices.emplace_back(index); }
        for (auto [index, _] : source) {
            if (std::find(indices.begin(), indices.end(), index) ==
                indices.end()) {
                indices.emplace_back(index);
            }
        }
        luisa::unordered_map<AllocaInst *, Set> merged;
        for (auto *index : indices) {
            auto lhs = target.contains(index) ?
                           target.find(index)->second :
                           old_feasible;
            auto rhs = source.contains(index) ?
                           source.find(index)->second :
                           incoming._feasible;
            auto unsafe = _manager->unite(lhs, rhs);
            if (unsafe != merged_feasible) {
                merged.emplace(index, unsafe);
            }
        }
        target = std::move(merged);
    };
    merge_relations(_less_unsafe, incoming._less_unsafe);
    merge_relations(_equal_unsafe, incoming._equal_unsafe);
    merge_relations(_last_unsafe, incoming._last_unsafe);
    merge_relations(_initialized_unsafe, incoming._initialized_unsafe);
    for (auto &[scalar, possibilities] :
         _masked_nonzero_possible) {
        auto incoming_scalar =
            incoming._masked_nonzero_possible.find(scalar);
        for (auto &[mask, possible] : possibilities) {
            auto incoming_possible =
                incoming_scalar !=
                        incoming._masked_nonzero_possible.end() &&
                    incoming_scalar->second.contains(mask) ?
                    incoming_scalar->second.find(mask)->second :
                    incoming._feasible;
            possible = _manager->unite(
                possible, incoming_possible);
        }
    }
    for (auto &[scalar, masks] :
         _masked_nonzero_positive_unsafe) {
        auto incoming_scalar =
            incoming._masked_nonzero_positive_unsafe.find(scalar);
        for (auto &[mask, unsafe] : masks) {
            auto incoming_unsafe =
                incoming_scalar !=
                        incoming._masked_nonzero_positive_unsafe.end() &&
                    incoming_scalar->second.contains(mask) ?
                    incoming_scalar->second.find(mask)->second :
                    incoming._feasible;
            unsafe = _manager->unite(unsafe, incoming_unsafe);
        }
    }
    for (auto &[scalar, possible] : _scalar_zero_possible) {
        auto incoming_possible =
            incoming._scalar_zero_possible.contains(scalar) ?
                incoming._scalar_zero_possible.find(scalar)->second :
                incoming._feasible;
        possible = _manager->unite(possible, incoming_possible);
    }
    for (auto &[scalar, unsafe] : _scalar_zero_positive_unsafe) {
        auto incoming_unsafe =
            incoming._scalar_zero_positive_unsafe.contains(scalar) ?
                incoming._scalar_zero_positive_unsafe.find(scalar)->second :
                incoming._feasible;
        unsafe = _manager->unite(unsafe, incoming_unsafe);
    }
    for (auto *index : incoming._tracked_indices) {
        _tracked_indices.emplace(index);
    }
    for (auto *predicate : incoming._tracked_predicates) {
        _tracked_predicates.emplace(predicate);
    }
    _feasible = merged_feasible;
    _counter_positive_unsafe = merged_counter_positive_unsafe;
    _tail_unsafe = merged_tail_unsafe;
    return *this != before;
}

}// namespace luisa::compute::xir::detail
