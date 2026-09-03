#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/verifier.h>

namespace luisa::compute::tile {

namespace detail {

struct ValueSlot {
    class CaptureContext *context{nullptr};
    Value *value{nullptr};
};

struct KernelStorage {
    luisa::unique_ptr<Module> module{luisa::make_unique<Module>()};
    Function *function{nullptr};
    Block *root{nullptr};
    luisa::vector<luisa::string> diagnostics;

    explicit KernelStorage(luisa::string_view name) noexcept {
        function = module->create_function(name);
        root = function->body().append_block();
    }
};

class CaptureContext final {

public:
    KernelStorage *kernel{nullptr};
    IRBuilder builder;
    luisa::vector<luisa::weak_ptr<ValueSlot>> slots;
    luisa::vector<ScopeStorage *> scopes;
    luisa::vector<std::pair<Dim, Value *>> coordinates;
    struct PositionalDimension {
        size_t from_right;
        Dim dimension;
    };
    luisa::vector<PositionalDimension> positional_dimensions;

    explicit CaptureContext(KernelStorage *kernel) noexcept
        : kernel{kernel}, builder{kernel->root} {}

    void error(luisa::string_view message) noexcept {
        kernel->diagnostics.emplace_back(message.data(), message.size());
    }

    [[nodiscard]] luisa::shared_ptr<ValueSlot> make_slot(Value *value) noexcept {
        auto slot = luisa::make_shared<ValueSlot>();
        slot->context = this;
        slot->value = value;
        slots.emplace_back(slot);
        return slot;
    }

    [[nodiscard]] Value *coordinate(Dim dimension, size_t visible_count) const noexcept {
        if (visible_count > coordinates.size()) { return nullptr; }
        for (auto i = visible_count; i != 0u; i--) {
            auto &&entry = coordinates[i - 1u];
            if (entry.first == dimension) { return entry.second; }
        }
        return nullptr;
    }
};

struct SlotSnapshot {
    luisa::weak_ptr<ValueSlot> slot;
    Value *value{nullptr};
    Operation *forwarder{nullptr};
    Value *replacement{nullptr};
};

struct ScopeStorage {
    CaptureContext *context{nullptr};
    const Nest *parent{nullptr};
    OperationKind kind{OperationKind::PARALLEL};
    IndexSpace domain;
    exec::Scope execution_scope{exec::Scope::AUTOMATIC};
    PipelinePolicy pipeline_policy;
    ScalarType element_type{ScalarType::INVALID};
    Value *element_result{nullptr};
    Block *parent_block{nullptr};
    Operation *operation{nullptr};
    Block *body{nullptr};
    size_t coordinate_base{0u};
    luisa::vector<SlotSnapshot> snapshot;
    Nest nest{this};
    bool entered{false};
    bool exited{false};
};

thread_local CaptureContext *current_capture = nullptr;

[[nodiscard]] luisa::string_view execution_scope_name(exec::Scope scope) noexcept {
    using namespace std::string_view_literals;
    switch (scope) {
        case exec::Scope::AUTOMATIC: return {};
        case exec::Scope::DEVICE: return "device"sv;
        case exec::Scope::GROUP: return "group"sv;
        case exec::Scope::SUBGROUP: return "subgroup"sv;
        case exec::Scope::WORKER: return "worker"sv;
        case exec::Scope::VECTOR: return "vector"sv;
    }
    return {};
}

void enter_scope(ScopeStorage &scope) noexcept {
    if (scope.entered || scope.exited) { return; }
    scope.entered = true;
    auto context = scope.context;
    if (context == nullptr || current_capture != context) { return; }
    if (!scope.domain.is_valid() || (scope.domain.empty() && scope.kind != OperationKind::TILE_MAP)) {
        context->error("execution nest requires a non-empty valid shape");
        return;
    }
    if (scope.parent != nullptr &&
        (context->scopes.empty() || &context->scopes.back()->nest != scope.parent)) {
        context->error("nested execution range must be created from the active parent nest");
    }
    if (scope.kind == OperationKind::PARALLEL && scope.execution_scope != exec::Scope::AUTOMATIC) {
        for (auto iter = context->scopes.rbegin(); iter != context->scopes.rend(); iter++) {
            if ((*iter)->kind != OperationKind::PARALLEL ||
                (*iter)->execution_scope == exec::Scope::AUTOMATIC) { continue; }
            if (static_cast<uint8_t>(scope.execution_scope) <
                static_cast<uint8_t>((*iter)->execution_scope)) {
                context->error("nested concrete execution scopes violate the predefined containment order");
            }
            break;
        }
    }
    for (auto iter = context->slots.begin(); iter != context->slots.end();) {
        if (auto slot = iter->lock()) {
            scope.snapshot.emplace_back(SlotSnapshot{slot, slot->value});
            iter++;
        } else {
            iter = context->slots.erase(iter);
        }
    }
    scope.parent_block = context->builder.insertion_block();
    scope.operation = scope.kind == OperationKind::TILE_MAP ?
                          context->builder.create_tile_map(Type::tile(scope.element_type, scope.domain)) :
                          context->builder.create_structured(scope.kind, scope.domain);
    if (scope.operation == nullptr) {
        context->error("failed to create structured TileIR operation");
        return;
    }
    if (scope.kind == OperationKind::PARALLEL) {
        auto name = execution_scope_name(scope.execution_scope);
        if (!name.empty()) { scope.operation->set_execution_scope_constraint(name); }
    }
    if (scope.kind == OperationKind::PIPELINE) {
        scope.operation->set_attribute("stages", Attribute{static_cast<uint64_t>(scope.pipeline_policy.stages)});
        scope.operation->set_attribute("initiation_interval", Attribute{static_cast<uint64_t>(scope.pipeline_policy.initiation_interval)});
    }
    scope.body = scope.operation->region(0u)->block(0u);
    scope.coordinate_base = context->coordinates.size();
    for (auto i = 0u; i < scope.domain.rank(); i++) {
        auto argument = scope.body->argument(i);
        auto dimension = scope.domain.axis(i).dimension;
        argument->set_name(context->kernel->module->dimensions().name(dimension));
        context->coordinates.emplace_back(dimension, argument);
    }
    context->scopes.emplace_back(&scope);
    context->builder.set_insertion_block(scope.body);
    if (scope.kind == OperationKind::SERIAL || scope.kind == OperationKind::PIPELINE || scope.kind == OperationKind::REDUCE) {
        // Distinguish variable identities before tracing the body. Copies may
        // share the same incoming SSA definition but diverge in later loop
        // iterations. Replacing that shared definition after capture would
        // incorrectly merge their recurrences (and immutable snapshots).
        // Same-type casts are temporary forwarding definitions, all removed
        // when the scope closes; they are not a new public IR primitive.
        for (auto &entry : scope.snapshot) {
            if (auto slot = entry.slot.lock(); slot != nullptr && entry.value != nullptr) {
                Value *operands[]{entry.value};
                entry.forwarder = context->builder.create_elementwise(ElementwiseOp::CAST, operands, entry.value->type());
                if (entry.forwarder == nullptr) {
                    context->error("failed to create a loop variable forwarding definition");
                    continue;
                }
                slot->value = entry.forwarder->result(0u);
            }
        }
    }
}

void exit_scope(ScopeStorage &scope) noexcept {
    if (!scope.entered || scope.exited) { return; }
    scope.exited = true;
    auto context = scope.context;
    if (context == nullptr || scope.operation == nullptr || scope.body == nullptr) { return; }
    if (context->scopes.empty() || context->scopes.back() != &scope) {
        context->error("execution nests must close in lexical order");
        return;
    }

    struct Carry {
        luisa::shared_ptr<ValueSlot> slot;
        Value *final{nullptr};
        Value *result{nullptr};
    };
    luisa::vector<Carry> carries;
    for (auto &entry : scope.snapshot) {
        entry.replacement = entry.value;
        auto slot = entry.slot.lock();
        auto incoming = entry.forwarder == nullptr ? entry.value : entry.forwarder->result(0u);
        if (!slot || slot->value == incoming) {
            if (slot) { slot->value = entry.value; }
            continue;
        }
        if (scope.kind == OperationKind::PARALLEL || scope.kind == OperationKind::TILE_MAP) {
            context->error("parallel nests and tile.map cannot mutate an outer value");
            slot->value = entry.value;
            continue;
        }
        if (entry.value == nullptr || slot->value == nullptr ||
            !(entry.value->type() == slot->value->type())) {
            context->error("loop-carried value assignment changed type or became invalid");
            slot->value = entry.value;
            continue;
        }
        auto final = slot->value;
        scope.operation->add_operand(entry.value);
        auto result = scope.operation->add_result(entry.value->type());
        auto argument = scope.body->add_argument(entry.value->type(), entry.value->name());
        entry.replacement = argument;
        carries.emplace_back(Carry{std::move(slot), final, result});
    }

    luisa::vector<Value *> yields;
    yields.reserve(carries.size());
    for (auto &&carry : carries) { yields.emplace_back(carry.final); }
    if (scope.kind == OperationKind::TILE_MAP) { yields.emplace_back(scope.element_result); }
    static_cast<void>(context->builder.create(OperationKind::YIELD, yields));
    // Install the yield first: a direct assignment such as b = old_a may
    // yield a forwarding definition itself, not an arithmetic use of it.
    luisa::unordered_map<Value *, Value *> replacements;
    for (auto &&entry : scope.snapshot) {
        if (entry.forwarder == nullptr) { continue; }
        auto forwarded = entry.forwarder->result(0u);
        if (!forwarded->replace_all_uses_with(entry.replacement)) {
            context->error("failed to resolve a loop variable forwarding definition");
            continue;
        }
        replacements.emplace(forwarded, entry.replacement);
    }
    // Live local C++ copies can also point directly at a forwarding value.
    // Rewrite all handles in one pass before deleting their IR definitions.
    for (auto &&weak : context->slots) {
        if (auto slot = weak.lock()) {
            if (auto iter = replacements.find(slot->value); iter != replacements.end()) { slot->value = iter->second; }
        }
    }
    for (auto &&entry : scope.snapshot) {
        if (entry.forwarder == nullptr || !replacements.contains(entry.forwarder->result(0u))) { continue; }
        if (!scope.body->erase(entry.forwarder)) { context->error("loop variable forwarding definition still has live uses"); }
    }
    context->builder.set_insertion_block(scope.parent_block);
    context->coordinates.resize(scope.coordinate_base);
    context->scopes.pop_back();
    for (auto &&carry : carries) { carry.slot->value = carry.result; }
}

ValueHandle::ValueHandle(luisa::shared_ptr<ValueSlot> slot) noexcept
    : _slot{std::move(slot)} {}

ValueHandle::ValueHandle(const ValueHandle &other) noexcept {
    if (other._slot != nullptr) {
        if (current_capture == nullptr || other._slot->context != current_capture) {
            capture_error("value copy requires the active capture");
        } else {
            _slot = current_capture->make_slot(other._slot->value);
        }
    }
}

ValueHandle::ValueHandle(ValueHandle &&other) noexcept = default;

ValueHandle &ValueHandle::operator=(const ValueHandle &other) noexcept {
    if (other._slot != nullptr && other._slot->context != current_capture) {
        capture_error("value assignment received a foreign or expired capture");
        return *this;
    }
    _assign(other.value());
    return *this;
}

ValueHandle &ValueHandle::operator=(ValueHandle &&other) noexcept {
    return operator=(static_cast<const ValueHandle &>(other));
}

ValueHandle::~ValueHandle() noexcept = default;

void ValueHandle::_assign(Value *value) noexcept {
    if (current_capture == nullptr) { return; }
    if (_slot == nullptr) {
        if (value != nullptr && current_capture != nullptr) { _slot = current_capture->make_slot(value); }
        return;
    }
    if (value == nullptr || _slot->context != current_capture || !(value->type() == _slot->value->type())) {
        capture_error("value assignment requires a valid value of the same type and capture");
        return;
    }
    _slot->value = value;
}

ValueHandle::operator bool() const noexcept {
    return current_capture != nullptr && _slot != nullptr &&
           _slot->context == current_capture && _slot->value != nullptr;
}

Value *ValueHandle::value() const noexcept {
    return static_cast<bool>(*this) ? _slot->value : nullptr;
}

ValueHandle make_constant(ScalarType type, Attribute value) noexcept {
    if (current_capture == nullptr || type == ScalarType::INVALID || !value.is_valid()) { return {}; }
    Type result_type[]{Type::scalar(type)};
    auto operation = current_capture->builder.create(OperationKind::CONSTANT, {}, result_type);
    if (operation == nullptr) {
        current_capture->error("failed to create scalar constant");
        return {};
    }
    operation->set_attribute("value", std::move(value));
    return ValueHandle{current_capture->make_slot(operation->result(0u))};
}

ValueHandle make_elementwise_operation(
    ElementwiseOp op,
    luisa::span<const ValueHandle> operands,
    ScalarType result_type) noexcept {
    if (current_capture == nullptr || op == ElementwiseOp::INVALID || result_type == ScalarType::INVALID) { return {}; }
    luisa::vector<Value *> values;
    values.reserve(operands.size());
    for (auto &&operand : operands) {
        if (!operand || operand._slot->context != current_capture) {
            current_capture->error("scalar operation received an invalid or foreign operand");
            return {};
        }
        values.emplace_back(operand.value());
    }
    auto operation = current_capture->builder.create_elementwise(op, values, Type::scalar(result_type));
    return operation == nullptr ? ValueHandle{} : ValueHandle{current_capture->make_slot(operation->result(0u))};
}

void capture_error(luisa::string_view message) noexcept {
    if (current_capture != nullptr) { current_capture->error(message); }
}

DeclaredMemory declare_memory(ScalarType type, const IndexSpace &space, mem::Resource resource, const IndexMap *layout) noexcept {
    if (current_capture == nullptr) { return {}; }
    luisa::string_view resource_name;
    switch (resource) {
        case mem::Resource::AUTOMATIC: break;
        case mem::Resource::PRIVATE: resource_name = "private"; break;
        case mem::Resource::SHARED: resource_name = "shared"; break;
        case mem::Resource::CLUSTER: resource_name = "cluster"; break;
        case mem::Resource::GLOBAL: resource_name = "global"; break;
        case mem::Resource::TENSOR: resource_name = "tensor"; break;
        default:
            capture_error("unknown Memory resource constraint");
            return {};
    }
    auto operation = layout == nullptr ?
                         current_capture->builder.create_memory_alloc(Type::memory(type, space), resource_name) :
                         current_capture->builder.create_memory_alloc(Type::memory(type, space), *layout, resource_name);
    if (operation == nullptr) {
        capture_error("failed to allocate Memory");
        return {};
    }
    return {operation->result(0u), ValueHandle{current_capture->make_slot(operation->result(1u))}};
}

IndexMap make_strided_layout(const IndexSpace &space, luisa::span<const uint64_t> strides) noexcept {
    if (current_capture == nullptr) { return {}; }
    auto storage = current_capture->kernel->module->dimensions().create_dimension("storage");
    auto result = IndexMap::strided(space, storage, strides);
    if (!result) {
        capture_error("layout requires static extents and one nonnegative element stride per logical axis");
        return {};
    }
    return std::move(*result);
}

ValueHandle load_memory(Value *memory, const ValueHandle &state) noexcept {
    if (current_capture == nullptr) { return {}; }
    if (!state || state._slot->context != current_capture || memory == nullptr) {
        capture_error("Memory load requires a live resource from the active capture");
        return {};
    }
    auto operation = current_capture->builder.create_memory_load(memory, state.value());
    if (operation == nullptr) {
        capture_error("failed to load Memory");
        return {};
    }
    return ValueHandle{current_capture->make_slot(operation->result(0u))};
}

void store_memory(Value *memory, ValueHandle &state, Value *tile) noexcept {
    if (current_capture == nullptr) { return; }
    if (!state || state._slot->context != current_capture || memory == nullptr || tile == nullptr ||
        tile->type() != memory->type().tile_value_type()) {
        capture_error("Memory store requires a live resource and an explicit Tile with the same element space and type");
        return;
    }
    auto operation = current_capture->builder.create_memory_store(memory, state.value(), tile);
    if (operation == nullptr) {
        capture_error("failed to store Memory");
        return;
    }
    state._assign(operation->result(0u));
}

ValueHandle make_tile_constant(ScalarType type, const IndexSpace &space, Attribute value) noexcept {
    if (current_capture == nullptr) { return {}; }
    Type result[]{Type::tile(type, space)};
    auto operation = current_capture->builder.create(OperationKind::CONSTANT, {}, result);
    if (operation == nullptr) { return {}; }
    operation->set_attribute("value", std::move(value));
    return ValueHandle{current_capture->make_slot(operation->result(0u))};
}

ValueHandle make_tile_elementwise(ElementwiseOp op, luisa::span<Value *const> operands, ScalarType type) noexcept {
    if (current_capture == nullptr) { return {}; }
    IndexSpace space;
    // Preserve the most informative operand's dimension order. For ite,
    // value operands take precedence over the boolean mask when ranks tie.
    auto first = op == ElementwiseOp::SELECT ? 1u : 0u;
    for (auto i = first; i < operands.size() + first; i++) {
        auto index = i % operands.size();
        auto value = operands[index];
        if (value != nullptr && value->type().is_tile() &&
            (space.empty() || value->type().index_space()->rank() > space.rank())) {
            space = *value->type().index_space();
        }
    }
    for (auto value : operands) {
        if (value == nullptr) {
            capture_error("Tile operation received an invalid operand");
            return {};
        }
        if (!value->type().is_tile()) { continue; }
        for (auto &&axis : value->type().index_space()->axes()) {
            if (auto existing = space.axis_index(axis.dimension)) {
                if (space.axis(*existing).extent != axis.extent) {
                    auto current = space.axis(*existing).extent;
                    if (axis.extent.is_constant() && axis.extent.constant_value() == 1u) { continue; }
                    if (!current.is_constant() || current.constant_value() != 1u) {
                        capture_error("Tile broadcasting requires equal or singleton extents for shared dimensions");
                        return {};
                    }
                    IndexSpace enlarged;
                    for (auto &&item : space.axes()) {
                        static_cast<void>(enlarged.add(item.dimension, item.dimension == axis.dimension ? axis.extent : item.extent));
                    }
                    space = std::move(enlarged);
                }
            } else if (!space.add(axis.dimension, axis.extent)) {
                capture_error("Tile broadcasting encountered an invalid dimension");
                return {};
            }
        }
    }
    auto operation = current_capture->builder.create_elementwise(op, operands, Type::tile(type, space));
    return operation == nullptr ? ValueHandle{} : ValueHandle{current_capture->make_slot(operation->result(0u))};
}

ValueHandle make_mma(Value *a, Value *b, Value *accumulator, MmaPolicy policy) noexcept {
    if (current_capture == nullptr) { return {}; }
    auto operation = current_capture->builder.create_mma(a, b, accumulator, policy);
    return operation == nullptr ? ValueHandle{} : ValueHandle{current_capture->make_slot(operation->result(0u))};
}

ValueHandle load_tile(Value *view, luisa::span<Value *const> origin, const IndexSpace &space,
                      BoundsMode bounds, Value *fallback) noexcept {
    if (current_capture == nullptr) { return {}; }
    auto operation = current_capture->builder.create_tile_load(view, origin, space, bounds, fallback);
    if (operation == nullptr) {
        capture_error("failed to create subtile load");
        return {};
    }
    return ValueHandle{current_capture->make_slot(operation->result(0u))};
}

void store_tile(Value *view, luisa::span<Value *const> origin, const IndexSpace &space,
                Value *tile, BoundsMode bounds) noexcept {
    if (current_capture != nullptr && current_capture->builder.create_tile_store(view, origin, space, tile, bounds) == nullptr) {
        capture_error("failed to create subtile store");
    }
}

ValueHandle extract_tile(Value *tile, luisa::span<Value *const> indices) noexcept {
    if (current_capture == nullptr) { return {}; }
    auto operation = current_capture->builder.create_tile_extract(tile, indices);
    if (operation == nullptr) {
        capture_error("failed to extract a Tile element");
        return {};
    }
    return ValueHandle{current_capture->make_slot(operation->result(0u))};
}

ValueHandle capture_tile_map(const IndexSpace &space, ScalarType type,
                             const std::function<Value *(const Nest &)> &body) noexcept {
    if (current_capture == nullptr) { return {}; }
    ScopeStorage scope;
    scope.context = current_capture;
    scope.kind = OperationKind::TILE_MAP;
    scope.domain = space;
    scope.element_type = type;
    enter_scope(scope);
    if (scope.operation == nullptr) { return {}; }
    scope.element_result = body(scope.nest);
    exit_scope(scope);
    return ValueHandle{current_capture->make_slot(scope.operation->result(0u))};
}

ValueHandle load_view(
    Value *view,
    luisa::span<const ValueHandle> indices,
    const ValueHandle *predicate,
    const ValueHandle *fallback) noexcept {
    if (current_capture == nullptr || view == nullptr || !view->type().is_view()) { return {}; }
    luisa::vector<Value *> values;
    values.reserve(indices.size());
    for (auto &&index : indices) {
        if (!index) {
            current_capture->error("view load received an invalid index");
            return {};
        }
        values.emplace_back(index.value());
    }
    if ((predicate == nullptr) != (fallback == nullptr) ||
        (predicate != nullptr && (!*predicate || !*fallback))) {
        current_capture->error("masked view load requires a valid predicate and fallback");
        return {};
    }
    auto operation = current_capture->builder.create_view_load(
        view,
        values,
        predicate == nullptr ? nullptr : predicate->value(),
        fallback == nullptr ? nullptr : fallback->value());
    return operation == nullptr ? ValueHandle{} : ValueHandle{current_capture->make_slot(operation->result(0u))};
}

void store_view(Value *view, luisa::span<const ValueHandle> indices, const ValueHandle &value) noexcept {
    if (current_capture == nullptr || view == nullptr || !view->type().is_view() || !value) { return; }
    luisa::vector<Value *> values;
    values.reserve(indices.size());
    for (auto &&index : indices) {
        if (!index) {
            current_capture->error("view store received an invalid index");
            return;
        }
        values.emplace_back(index.value());
    }
    if (current_capture->builder.create_view_store(view, values, value.value()) == nullptr) {
        current_capture->error("failed to create view store");
    }
}

CaptureGuard::CaptureGuard(Kernel &kernel) noexcept
    : _kernel{&kernel} {
    if (current_capture != nullptr) {
        kernel._storage->diagnostics.emplace_back("Tile kernel captures cannot be nested");
        return;
    }
    current_capture = new CaptureContext{kernel._storage.get()};
}

CaptureGuard::~CaptureGuard() noexcept {
    if (_kernel == nullptr || current_capture == nullptr || current_capture->kernel != _kernel->_storage.get()) { return; }
    while (!current_capture->scopes.empty()) {
        current_capture->error("unterminated execution nest at end of kernel capture");
        exit_scope(*current_capture->scopes.back());
    }
    // Escaped C++ handles must not retain a dangling context pointer that a
    // later capture could reuse at the same address. Their IR values still
    // belong to the captured Module, but cannot participate in a new trace.
    for (auto &&weak : current_capture->slots) {
        if (auto slot = weak.lock()) { slot->context = nullptr; }
    }
    delete current_capture;
    current_capture = nullptr;
}

Axis create_axis(luisa::string_view name, Extent extent) noexcept {
    if (current_capture == nullptr || !extent.is_valid()) { return {}; }
    auto dimension = current_capture->kernel->module->dimensions().create_dimension(name);
    return Axis{dimension, extent};
}

IndexSpace make_shape(luisa::span<const Axis> axes) noexcept {
    IndexSpace result;
    if (current_capture == nullptr || axes.empty()) { return result; }
    for (auto &&axis : axes) {
        if (!axis || axis.dimension().context() != &current_capture->kernel->module->dimensions() ||
            !result.add(axis.dimension(), axis.extent())) {
            current_capture->error("shape axes must be valid, unique, and belong to the active kernel");
            return {};
        }
    }
    return result;
}

IndexSpace make_positional_shape(luisa::span<const uint64_t> extents) noexcept {
    IndexSpace result;
    if (current_capture == nullptr) { return result; }
    for (auto i = 0u; i < extents.size(); i++) {
        auto from_right = extents.size() - i - 1u;
        Dim dimension;
        for (auto &&candidate : current_capture->positional_dimensions) {
            if (candidate.from_right == from_right) {
                dimension = candidate.dimension;
                break;
            }
        }
        if (!dimension) {
            dimension = current_capture->kernel->module->dimensions().create_dimension(
                luisa::format("positional.{}", from_right));
            current_capture->positional_dimensions.emplace_back(CaptureContext::PositionalDimension{from_right, dimension});
        }
        static_cast<void>(result.add(dimension, extents[i]));
    }
    return result;
}

DeclaredTensorView declare_tensor_view(
    size_t argument_index,
    luisa::string_view name,
    ScalarType element_type,
    luisa::span<const uint64_t> extents) noexcept {
    if (current_capture == nullptr || extents.empty()) { return {}; }
    auto root = current_capture->kernel->root;
    if (current_capture->builder.insertion_block() != root ||
        root->argument_count() != argument_index) {
        current_capture->error("kernel parameters must be created in signature order before the body");
        return {};
    }
    auto argument_name = name.empty() ?
                             luisa::format("arg{}", argument_index) :
                             luisa::string{name};
    for (auto &&argument : root->arguments()) {
        if (argument->name() == argument_name) {
            current_capture->error("kernel parameter names must be unique");
            return {};
        }
    }
    luisa::vector<Axis> axes;
    axes.reserve(extents.size());
    for (auto i = 0u; i < extents.size(); i++) {
        axes.emplace_back(create_axis(
            luisa::format("{}.{}", argument_name, i), Extent::constant(extents[i])));
    }
    auto space = make_shape(axes);
    auto view = root->add_argument(Type::view(element_type, space), argument_name);
    return DeclaredTensorView{view, std::move(space)};
}

luisa::vector<ValueHandle> nest_indices(const Nest &nest, const IndexSpace &space) noexcept {
    luisa::vector<ValueHandle> result;
    if (current_capture == nullptr || nest._scope == nullptr || nest._scope->context != current_capture) { return result; }
    result.reserve(space.rank());
    for (auto &&axis : space.axes()) {
        auto coordinate = current_capture->coordinate(
            axis.dimension, nest._scope->coordinate_base + nest._scope->domain.rank());
        if (coordinate == nullptr) {
            current_capture->error("active execution hierarchy does not define every requested View dimension");
            return {};
        }
        result.emplace_back(ValueHandle{current_capture->make_slot(coordinate)});
    }
    return result;
}

NestRange make_range(
    const Nest *parent,
    OperationKind kind,
    IndexSpace domain,
    exec::Scope scope,
    PipelinePolicy policy) noexcept {
    if (current_capture == nullptr) { return NestRange{nullptr}; }
    auto storage = luisa::make_unique<ScopeStorage>();
    storage->context = current_capture;
    storage->parent = parent;
    storage->kind = kind;
    storage->domain = std::move(domain);
    storage->execution_scope = scope;
    storage->pipeline_policy = policy;
    return NestRange{std::move(storage)};
}

}// namespace detail

Kernel::Kernel(luisa::string_view name) noexcept
    : _storage{luisa::make_unique<detail::KernelStorage>(name)} {}

Kernel::Kernel(Kernel &&) noexcept = default;
Kernel &Kernel::operator=(Kernel &&) noexcept = default;
Kernel::~Kernel() noexcept = default;

Module &Kernel::module() noexcept { return *_storage->module; }
const Module &Kernel::module() const noexcept { return *_storage->module; }
Function &Kernel::function() noexcept { return *_storage->function; }
const Function &Kernel::function() const noexcept { return *_storage->function; }

bool Kernel::valid() const noexcept {
    return _storage != nullptr && _storage->diagnostics.empty() && verify(*_storage->module).ok();
}

luisa::span<const luisa::string> Kernel::diagnostics() const noexcept {
    return _storage == nullptr ? luisa::span<const luisa::string>{} : luisa::span<const luisa::string>{_storage->diagnostics};
}

Scalar<int64_t> Nest::index(const Axis &axis) const noexcept {
    return index(axis.dimension());
}

Scalar<int64_t> Nest::index(Dim dimension) const noexcept {
    if (_scope == nullptr || _scope->context == nullptr || !dimension) { return {}; }
    // Resolve against this Nest and its ancestors, not a currently active
    // descendant that may reuse the same dimension identity.
    auto value = _scope->context->coordinate(dimension, _scope->coordinate_base + _scope->domain.rank());
    if (value == nullptr) { _scope->context->error("the active hierarchy does not define this coordinate"); }
    return value == nullptr ? Scalar<int64_t>{} : Scalar<int64_t>{detail::ValueHandle{_scope->context->make_slot(value)}};
}

Scalar<int64_t> Nest::index() const noexcept {
    if (_scope == nullptr || _scope->domain.rank() != 1u) {
        if (_scope != nullptr) { _scope->context->error("index() without an Axis requires a rank-one local nest"); }
        return {};
    }
    return index(_scope->domain.axis(0u).dimension);
}

NestRange Nest::parallel(IndexSpace domain, exec::Scope scope) const noexcept {
    return detail::make_range(this, OperationKind::PARALLEL, std::move(domain), scope, {});
}

NestRange Nest::serial(IndexSpace domain) const noexcept {
    return detail::make_range(this, OperationKind::SERIAL, std::move(domain), exec::Scope::AUTOMATIC, {});
}

NestRange Nest::reduce(IndexSpace domain) const noexcept {
    return detail::make_range(this, OperationKind::REDUCE, std::move(domain), exec::Scope::AUTOMATIC, {});
}

NestRange Nest::pipeline(IndexSpace domain, PipelinePolicy policy) const noexcept {
    return detail::make_range(this, OperationKind::PIPELINE, std::move(domain), exec::Scope::AUTOMATIC, policy);
}

void Nest::stage(luisa::string_view name) const noexcept {
    if (_scope == nullptr || _scope->context == nullptr || _scope->kind != OperationKind::PIPELINE ||
        _scope->context->scopes.empty() || _scope->context->scopes.back() != _scope) {
        if (_scope != nullptr && _scope->context != nullptr) {
            _scope->context->error("stage() is only valid on the active pipeline nest");
        }
        return;
    }
    auto operation = _scope->context->builder.create(OperationKind::STAGE);
    if (operation != nullptr && !name.empty()) {
        operation->set_attribute("name", Attribute{name});
    }
}

NestRange::NestRange(luisa::unique_ptr<detail::ScopeStorage> storage) noexcept
    : _storage{std::move(storage)} {}

NestRange::NestRange(NestRange &&) noexcept = default;
NestRange &NestRange::operator=(NestRange &&) noexcept = default;

NestRange::~NestRange() noexcept {
    _exit();
}

void NestRange::_enter() noexcept {
    if (_storage != nullptr) { detail::enter_scope(*_storage); }
}

void NestRange::_exit() noexcept {
    if (_storage != nullptr) { detail::exit_scope(*_storage); }
}

Nest &NestRange::_nest() noexcept { return _storage->nest; }

Nest &NestIterator::operator*() const noexcept { return _range->_nest(); }

NestIterator &NestIterator::operator++() noexcept {
    if (!_done) {
        _range->_exit();
        _done = true;
    }
    return *this;
}

}// namespace luisa::compute::tile
