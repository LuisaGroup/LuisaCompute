#include <algorithm>

#include <luisa/tile/ir.h>

namespace luisa::compute::tile {

Type::Type(TypeKind kind, ScalarType scalar, const IndexSpace &space) noexcept
    : _kind{kind}, _scalar{scalar}, _space{luisa::make_shared<IndexSpace>(space)} {}

Type Type::scalar(ScalarType scalar) noexcept {
    Type type{TypeKind::SCALAR};
    type._scalar = scalar;
    return type;
}

Type Type::tile(ScalarType scalar, const IndexSpace &space) noexcept {
    return Type{TypeKind::TILE, scalar, space};
}

Type Type::memory(ScalarType scalar, const IndexSpace &space) noexcept {
    return Type{TypeKind::MEMORY, scalar, space};
}

Type Type::opaque(luisa::string_view name) noexcept {
    Type type{TypeKind::OPAQUE};
    type._opaque_name = luisa::string{name.data(), name.size()};
    return type;
}

bool Type::is_valid() const noexcept {
    switch (_kind) {
        case TypeKind::INVALID: return false;
        case TypeKind::INDEX:
        case TypeKind::MEMORY_STATE: return true;
        case TypeKind::SCALAR: return _scalar != ScalarType::INVALID;
        case TypeKind::TILE:
        case TypeKind::MEMORY: return _scalar != ScalarType::INVALID && _space != nullptr && _space->is_valid();
        case TypeKind::OPAQUE: return !_opaque_name.empty();
    }
    return false;
}

Type Type::tile_value_type() const noexcept {
    return is_memory() && _space != nullptr ? Type::tile(_scalar, *_space) : Type{};
}

bool operator==(const Type &lhs, const Type &rhs) noexcept {
    if (lhs._kind != rhs._kind || lhs._scalar != rhs._scalar || lhs._opaque_name != rhs._opaque_name) { return false; }
    if (lhs._space == rhs._space) { return true; }
    return lhs._space != nullptr && rhs._space != nullptr && *lhs._space == *rhs._space;
}

luisa::string_view to_string(OperationKind kind) noexcept {
    using namespace std::string_view_literals;
    switch (kind) {
        case OperationKind::CUSTOM: return "custom"sv;
        case OperationKind::CONSTANT: return "tile.constant"sv;
        case OperationKind::TILE_MAP: return "tile.map"sv;
        case OperationKind::MMA: return "tile.mma"sv;
        case OperationKind::VIEW_LOAD: return "tile.view.load"sv;
        case OperationKind::VIEW_STORE: return "tile.view.store"sv;
        case OperationKind::MEMORY_ALLOC: return "tile.memory.alloc"sv;
        case OperationKind::MEMORY_LOAD: return "tile.memory.load"sv;
        case OperationKind::MEMORY_STORE: return "tile.memory.store"sv;
        case OperationKind::PARALLEL: return "tile.parallel"sv;
        case OperationKind::SERIAL: return "tile.serial"sv;
        case OperationKind::PIPELINE: return "tile.pipeline"sv;
        case OperationKind::REDUCE: return "tile.reduce"sv;
        case OperationKind::YIELD: return "tile.yield"sv;
    }
    return "tile.unknown"sv;
}

Value::Value(uint64_t id, Type type, Block *block, size_t index) noexcept
    : _id{id}, _type{std::move(type)}, _origin{Origin::BLOCK_ARGUMENT}, _block{block}, _index{index} {}

Value::Value(uint64_t id, Type type, Operation *operation, size_t index) noexcept
    : _id{id}, _type{std::move(type)}, _origin{Origin::OPERATION_RESULT}, _operation{operation}, _index{index} {}

void Value::_add_use(Use *use) noexcept {
    if (use != nullptr && std::find(_uses.begin(), _uses.end(), use) == _uses.end()) { _uses.emplace_back(use); }
}

void Value::_remove_use(Use *use) noexcept {
    if (auto iter = std::find(_uses.begin(), _uses.end(), use); iter != _uses.end()) { _uses.erase(iter); }
}

Value::~Value() noexcept {
    while (!_uses.empty()) {
        auto use = _uses.back();
        _uses.pop_back();
        use->_value = nullptr;
    }
}

bool Value::replace_all_uses_with(Value *replacement) noexcept {
    if (replacement == nullptr || replacement == this || !(replacement->type() == type())) { return false; }
    while (!_uses.empty()) { _uses.back()->set(replacement); }
    return true;
}

Use::Use(Operation *user, size_t index, Value *value) noexcept
    : _user{user}, _index{index} {
    set(value);
}

Use::~Use() noexcept {
    set(nullptr);
}

void Use::set(Value *value) noexcept {
    if (_value == value) { return; }
    if (_value != nullptr) { _value->_remove_use(this); }
    _value = value;
    if (_value != nullptr) { _value->_add_use(this); }
}

Operation::Operation(uint64_t id, Block *parent, OperationKind kind, luisa::string_view custom_name) noexcept
    : _id{id}, _parent{parent}, _kind{kind}, _custom_name{custom_name.data(), custom_name.size()} {}

Operation::~Operation() noexcept {
    _regions.clear();
    _operands.clear();
    _results.clear();
}

luisa::string_view Operation::name() const noexcept {
    return _kind == OperationKind::CUSTOM ? luisa::string_view{_custom_name} : to_string(_kind);
}

Function *Operation::parent_function() noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

const Function *Operation::parent_function() const noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

MemoryEffect Operation::memory_effect() const noexcept {
    switch (_kind) {
        case OperationKind::VIEW_LOAD:
        case OperationKind::MEMORY_LOAD: return MemoryEffect::READ;
        case OperationKind::VIEW_STORE:
        case OperationKind::MEMORY_STORE: return MemoryEffect::WRITE;
        case OperationKind::MEMORY_ALLOC: return MemoryEffect::ALLOCATE;
        case OperationKind::CUSTOM: return MemoryEffect::UNKNOWN;
        default: return MemoryEffect::NONE;
    }
}

void Operation::add_operand(Value *value) noexcept {
    _operands.emplace_back(luisa::make_unique<Use>(this, _operands.size(), value));
}

void Operation::set_operand(size_t index, Value *value) noexcept {
    _operands[index]->set(value);
}

Value *Operation::add_result(Type type) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto value = luisa::unique_ptr<Value>{new Value{function->_allocate_value_id(), std::move(type), this, _results.size()}};
    auto result = value.get();
    _results.emplace_back(std::move(value));
    return result;
}

Region *Operation::add_region(luisa::string_view label) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto region = luisa::unique_ptr<Region>{new Region{function, this, label}};
    auto result = region.get();
    _regions.emplace_back(std::move(region));
    return result;
}

void Operation::set_execution_scope_constraint(luisa::string_view scope) noexcept {
    _execution_scope_constraint = scope.empty() ? luisa::optional<luisa::string>{} : luisa::optional<luisa::string>{luisa::string{scope.data(), scope.size()}};
}

void Operation::set_resource_class_constraint(luisa::string_view resource) noexcept {
    _resource_class_constraint = resource.empty() ? luisa::optional<luisa::string>{} : luisa::optional<luisa::string>{luisa::string{resource.data(), resource.size()}};
}

void Operation::set_attribute(luisa::string_view name, Attribute value) noexcept {
    for (auto &&attribute : _attributes) {
        if (luisa::string_view{attribute.name} == name) {
            attribute.value = std::move(value);
            return;
        }
    }
    _attributes.emplace_back(NamedAttribute{luisa::string{name.data(), name.size()}, std::move(value)});
}

const Attribute *Operation::attribute(luisa::string_view name) const noexcept {
    for (auto &&attribute : _attributes) {
        if (luisa::string_view{attribute.name} == name) { return &attribute.value; }
    }
    return nullptr;
}

Block::~Block() noexcept {
    _operations.clear();
    _arguments.clear();
}

Function *Block::parent_function() noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

const Function *Block::parent_function() const noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

Value *Block::add_argument(Type type) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto argument = luisa::unique_ptr<Value>{new Value{function->_allocate_value_id(), std::move(type), this, _arguments.size()}};
    auto result = argument.get();
    _arguments.emplace_back(std::move(argument));
    return result;
}

Operation *Block::append_operation(OperationKind kind, luisa::string_view custom_name) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto operation = luisa::unique_ptr<Operation>{new Operation{function->_allocate_operation_id(), this, kind, custom_name}};
    auto result = operation.get();
    _operations.emplace_back(std::move(operation));
    return result;
}

bool Block::erase(Operation *operation) noexcept {
    auto iter = std::find_if(_operations.begin(), _operations.end(), [operation](auto &&item) noexcept { return item.get() == operation; });
    if (iter == _operations.end()) { return false; }
    for (auto i = 0u; i < operation->result_count(); i++) {
        if (operation->result(i)->use_count() != 0u) { return false; }
    }
    _operations.erase(iter);
    return true;
}

Region::Region(Function *function, Operation *parent, luisa::string_view label) noexcept
    : _function{function}, _parent{parent}, _label{label.data(), label.size()} {}

Region::~Region() noexcept = default;

Block *Region::append_block() noexcept {
    auto block = luisa::unique_ptr<Block>{new Block{this}};
    auto result = block.get();
    _blocks.emplace_back(std::move(block));
    return result;
}

Function::Function(Module *parent, uint64_t id, luisa::string_view name, IRForm form) noexcept
    : _parent{parent}, _id{id}, _name{name.data(), name.size()}, _form{form}, _body{this, nullptr, "body"} {}

Function::~Function() noexcept = default;

Module::~Module() noexcept = default;

Function *Module::create_function(luisa::string_view name, IRForm form) noexcept {
    auto function = luisa::unique_ptr<Function>{new Function{this, _next_function_id++, name, form}};
    auto result = function.get();
    _functions.emplace_back(std::move(function));
    return result;
}

Operation *IRBuilder::create(OperationKind kind,
                             luisa::span<Value *const> operands,
                             luisa::span<const Type> result_types,
                             luisa::string_view custom_name) noexcept {
    if (_insertion_block == nullptr) { return nullptr; }
    auto operation = _insertion_block->append_operation(kind, custom_name);
    for (auto operand : operands) { operation->add_operand(operand); }
    for (auto &&type : result_types) { static_cast<void>(operation->add_result(type)); }
    return operation;
}

Operation *IRBuilder::create_structured(OperationKind kind,
                                        IndexSpace domain,
                                        luisa::span<const luisa::string_view> region_labels,
                                        luisa::span<const Type> result_types) noexcept {
    if (kind != OperationKind::PARALLEL && kind != OperationKind::SERIAL &&
        kind != OperationKind::PIPELINE && kind != OperationKind::REDUCE) { return nullptr; }
    auto operation = create(kind, {}, result_types);
    if (operation == nullptr) { return nullptr; }
    operation->set_domain(std::move(domain));
    auto region_count = region_labels.empty() ? 1u : region_labels.size();
    for (auto i = 0u; i < region_count; i++) {
        auto label = region_labels.empty() ? luisa::string_view{} : region_labels[i];
        auto block = operation->add_region(label)->append_block();
        for (auto j = 0u; j < operation->domain()->rank(); j++) { static_cast<void>(block->add_argument(Type::index())); }
    }
    return operation;
}

Operation *IRBuilder::create_mma(Value *a, Value *b, Value *accumulator) noexcept {
    if (a == nullptr || b == nullptr || accumulator == nullptr) { return nullptr; }
    Value *operands[]{a, b, accumulator};
    Type results[]{accumulator->type()};
    return create(OperationKind::MMA, operands, results);
}

Operation *IRBuilder::create_memory_alloc(const Type &memory_type, luisa::string_view resource_class) noexcept {
    if (!memory_type.is_memory()) { return nullptr; }
    Type results[]{memory_type, Type::memory_state()};
    auto operation = create(OperationKind::MEMORY_ALLOC, {}, results);
    if (operation != nullptr) { operation->set_resource_class_constraint(resource_class); }
    return operation;
}

Operation *IRBuilder::create_memory_load(Value *memory, Value *state) noexcept {
    if (memory == nullptr || state == nullptr || !memory->type().is_memory()) { return nullptr; }
    Value *operands[]{memory, state};
    Type results[]{memory->type().tile_value_type()};
    return create(OperationKind::MEMORY_LOAD, operands, results);
}

Operation *IRBuilder::create_memory_store(Value *memory, Value *state, Value *tile) noexcept {
    if (memory == nullptr || state == nullptr || tile == nullptr) { return nullptr; }
    Value *operands[]{memory, state, tile};
    Type results[]{Type::memory_state()};
    return create(OperationKind::MEMORY_STORE, operands, results);
}

}// namespace luisa::compute::tile
