#include <luisa/core/logging.h>
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

Type Type::view(ScalarType scalar, const IndexSpace &space) noexcept {
    return Type{TypeKind::VIEW, scalar, space};
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
        case TypeKind::VIEW:
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
        case OperationKind::ELEMENTWISE: return "tile.elementwise"sv;
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
        case OperationKind::STAGE: return "tile.stage"sv;
        case OperationKind::REDUCE: return "tile.reduce"sv;
        case OperationKind::YIELD: return "tile.yield"sv;
    }
    return "tile.unknown"sv;
}

luisa::string_view to_string(ElementwiseOp op) noexcept {
    using namespace std::string_view_literals;
    switch (op) {
        case ElementwiseOp::INVALID: return "invalid"sv;
        case ElementwiseOp::ADD: return "add"sv;
        case ElementwiseOp::SUB: return "sub"sv;
        case ElementwiseOp::MUL: return "mul"sv;
        case ElementwiseOp::DIV: return "div"sv;
        case ElementwiseOp::MOD: return "mod"sv;
        case ElementwiseOp::NEG: return "neg"sv;
        case ElementwiseOp::MIN: return "min"sv;
        case ElementwiseOp::MAX: return "max"sv;
        case ElementwiseOp::CAST: return "cast"sv;
        case ElementwiseOp::SELECT: return "select"sv;
        case ElementwiseOp::EQ: return "eq"sv;
        case ElementwiseOp::NE: return "ne"sv;
        case ElementwiseOp::LT: return "lt"sv;
        case ElementwiseOp::LE: return "le"sv;
        case ElementwiseOp::GT: return "gt"sv;
        case ElementwiseOp::GE: return "ge"sv;
        case ElementwiseOp::EXP: return "exp"sv;
        case ElementwiseOp::LOG: return "log"sv;
        case ElementwiseOp::SQRT: return "sqrt"sv;
        case ElementwiseOp::TANH: return "tanh"sv;
        case ElementwiseOp::ABS: return "abs"sv;
    }
    return "invalid"sv;
}

Value::Value(uint64_t id, Type type, Block *block, size_t index) noexcept
    : _id{id}, _type{std::move(type)}, _origin{Origin::BLOCK_ARGUMENT}, _block{block}, _index{index} {}

Value::Value(uint64_t id, Type type, Operation *operation, size_t index) noexcept
    : _id{id}, _type{std::move(type)}, _origin{Origin::OPERATION_RESULT}, _operation{operation}, _index{index} {}

Use::Use(Operation *user, size_t index) noexcept
    : _user{user}, _index{index} {
    LUISA_DEBUG_ASSERT(user != nullptr, "Use requires a non-null user.");
}

Use *UseList::push_front(luisa::ManagedPtr<Use> use) noexcept {
    LUISA_DEBUG_ASSERT(use != nullptr && use->_list_owner == nullptr && !use->is_linked(),
                       "Use is already linked to an owner list.");
    auto node = _nodes.push_front(std::move(use));
    node->_list_owner = this;
    return node;
}

luisa::ManagedPtr<Use> Use::remove_self() noexcept {
    auto was_linked = is_linked();
    LUISA_DEBUG_ASSERT(was_linked == (_list_owner != nullptr),
                       "Use linkage and owner-list identity disagree.");
    auto self = Super::remove_self();
    if (self != nullptr) {
        LUISA_DEBUG_ASSERT(was_linked && self.get() == this,
                           "Removed Use ownership is inconsistent.");
        _list_owner = nullptr;
    }
    return self;
}

void Use::set(Value *value) noexcept {
    if (_value == value) { return; }
    auto owned = is_linked() ? remove_self() : lock();
    _value = value;
    if (_value != nullptr && _user->is_linked()) {
        _value->use_list().push_front(std::move(owned));
    }
}

Value::~Value() noexcept {
    while (!_use_list.empty()) { _use_list.front()->set(nullptr); }
}

bool Value::replace_all_uses_with(Value *replacement) noexcept {
    if (replacement == nullptr || replacement == this || !(replacement->type() == type())) { return false; }
    while (!_use_list.empty()) { _use_list.front()->set(replacement); }
    return true;
}

void Operation::_remove_self_from_operand_use_lists() noexcept {
    for (auto &&use : _operands) {
        if (use->is_linked()) { static_cast<void>(use->remove_self()); }
    }
}

void Operation::_add_self_to_operand_use_lists() noexcept {
    LUISA_DEBUG_ASSERT(is_linked(), "Cannot attach operands for a detached Operation.");
    for (auto &&use : _operands) {
        LUISA_DEBUG_ASSERT(!use->is_linked(), "Operand Use is already linked.");
        if (auto value = use->value()) { value->use_list().push_front(use->lock()); }
    }
}

Operation::Operation(uint64_t id, Block *parent, OperationKind kind, luisa::string_view custom_name) noexcept
    : _id{id}, _parent{parent}, _kind{kind}, _custom_name{custom_name.data(), custom_name.size()} {}

Operation::~Operation() noexcept {
    _remove_self_from_operand_use_lists();
    _regions.clear();
    _operands.clear();
    _results.clear();
}

luisa::ManagedPtr<Operation> Operation::remove_self() noexcept {
    if (!is_linked()) { return nullptr; }
    _remove_self_from_operand_use_lists();
    return Super::remove_self();
}

Operation *Operation::insert_before_self(luisa::ManagedPtr<Operation> operation) noexcept {
    auto inserted = Super::insert_before_self(std::move(operation));
    inserted->_parent = _parent;
    inserted->_add_self_to_operand_use_lists();
    return inserted;
}

SentinelOperation::SentinelOperation(Block *parent) noexcept
    : Operation{~0ull, parent, OperationKind::CUSTOM} {}

luisa::string_view Operation::name() const noexcept {
    if (_kind == OperationKind::CUSTOM) { return _custom_name; }
    if (_kind == OperationKind::ELEMENTWISE) { return to_string(_elementwise_op); }
    return to_string(_kind);
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
    auto use = luisa::make_managed<Use>(this, _operands.size());
    auto ptr = use.get();
    _operands.emplace_back(std::move(use));
    ptr->set(value);
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

Block::Block(Region *parent) noexcept
    : _parent{parent}, _operations{this} {}

Block::~Block() noexcept = default;

luisa::ManagedPtr<Block> Block::remove_self() noexcept {
    return Super::remove_self();
}

Block *Block::insert_before_self(luisa::ManagedPtr<Block> block) noexcept {
    auto inserted = Super::insert_before_self(std::move(block));
    inserted->_parent = _parent;
    return inserted;
}

SentinelBlock::SentinelBlock(Region *parent) noexcept
    : Block{parent} {}

Function *Block::parent_function() noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

const Function *Block::parent_function() const noexcept {
    return _parent == nullptr ? nullptr : _parent->parent_function();
}

Value *Block::add_argument(Type type, luisa::string_view name) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto argument = luisa::unique_ptr<Value>{new Value{function->_allocate_value_id(), std::move(type), this, _arguments.size()}};
    auto result = argument.get();
    result->set_name(name);
    _arguments.emplace_back(std::move(argument));
    return result;
}

Operation *Block::append_operation(OperationKind kind, luisa::string_view custom_name) noexcept {
    auto function = parent_function();
    if (function == nullptr) { return nullptr; }
    auto operation = luisa::make_managed<Operation>(
        function->_allocate_operation_id(), this, kind, custom_name);
    return _operations.push_back(std::move(operation));
}

Operation *Block::insert_operation_before(
    Operation *position,
    OperationKind kind,
    luisa::string_view custom_name) noexcept {
    auto function = parent_function();
    if (function == nullptr || position == nullptr || position->parent_block() != this ||
        !position->is_linked()) { return nullptr; }
    auto operation = luisa::make_managed<Operation>(
        function->_allocate_operation_id(), this, kind, custom_name);
    return position->insert_before_self(std::move(operation));
}

bool Block::erase(Operation *operation) noexcept {
    if (operation == nullptr || operation->parent_block() != this || !operation->is_linked() || operation->is_sentinel()) { return false; }
    for (auto i = 0u; i < operation->result_count(); i++) {
        if (operation->result(i)->use_count() != 0u) { return false; }
    }
    return operation->remove_self() != nullptr;
}

Operation *Block::operation(size_t index) noexcept {
    auto i = 0u;
    for (auto operation : _operations) {
        if (i++ == index) { return operation; }
    }
    return nullptr;
}

const Operation *Block::operation(size_t index) const noexcept {
    return const_cast<Block *>(this)->operation(index);
}

Region::Region(Function *function, Operation *parent, luisa::string_view label) noexcept
    : _function{function}, _parent{parent}, _label{label.data(), label.size()}, _blocks{this} {}

Region::~Region() noexcept = default;

Block *Region::append_block() noexcept {
    return _blocks.push_back(luisa::make_managed<Block>(this));
}

Block *Region::block(size_t index) noexcept {
    auto i = 0u;
    for (auto block : _blocks) {
        if (i++ == index) { return block; }
    }
    return nullptr;
}

const Block *Region::block(size_t index) const noexcept {
    return const_cast<Region *>(this)->block(index);
}

Function::Function(Module *parent, uint64_t id, luisa::string_view name, IRForm form) noexcept
    : _parent{parent}, _id{id}, _name{name.data(), name.size()}, _form{form}, _body{this, nullptr, "body"} {}

Function::~Function() noexcept = default;

luisa::ManagedPtr<Function> Function::remove_self() noexcept {
    return Super::remove_self();
}

Function *Function::insert_before_self(luisa::ManagedPtr<Function> function) noexcept {
    auto inserted = Super::insert_before_self(std::move(function));
    inserted->_parent = _parent;
    return inserted;
}

SentinelFunction::SentinelFunction(Module *parent) noexcept
    : Function{parent, ~0ull, {}, IRForm::CANDIDATE} {}

Module::Module() noexcept
    : _functions{this} {}

Module::~Module() noexcept = default;

Function *Module::create_function(luisa::string_view name, IRForm form) noexcept {
    return _functions.push_back(luisa::make_managed<Function>(this, _next_function_id++, name, form));
}

Function *Module::function(size_t index) noexcept {
    auto i = 0u;
    for (auto function : _functions) {
        if (i++ == index) { return function; }
    }
    return nullptr;
}

const Function *Module::function(size_t index) const noexcept {
    return const_cast<Module *>(this)->function(index);
}

Operation *IRBuilder::create(OperationKind kind,
                             luisa::span<Value *const> operands,
                             luisa::span<const Type> result_types,
                             luisa::string_view custom_name) noexcept {
    if (_insertion_block == nullptr) { return nullptr; }
    auto operation = _insertion_before == nullptr ?
                         _insertion_block->append_operation(kind, custom_name) :
                         _insertion_block->insert_operation_before(_insertion_before, kind, custom_name);
    for (auto operand : operands) { operation->add_operand(operand); }
    for (auto &&type : result_types) { static_cast<void>(operation->add_result(type)); }
    return operation;
}

Operation *IRBuilder::create_structured(OperationKind kind,
                                        IndexSpace domain,
                                        luisa::span<Value *const> operands,
                                        luisa::span<const Type> result_types) noexcept {
    if (kind != OperationKind::PARALLEL && kind != OperationKind::SERIAL &&
        kind != OperationKind::PIPELINE && kind != OperationKind::REDUCE) { return nullptr; }
    auto operation = create(kind, operands, result_types);
    if (operation == nullptr) { return nullptr; }
    operation->set_domain(std::move(domain));
    auto block = operation->add_region("body")->append_block();
    for (auto i = 0u; i < operation->domain()->rank(); i++) {
        static_cast<void>(block->add_argument(Type::index()));
    }
    for (auto &&type : result_types) { static_cast<void>(block->add_argument(type)); }
    return operation;
}

Operation *IRBuilder::create_elementwise(
    ElementwiseOp op,
    luisa::span<Value *const> operands,
    Type result_type) noexcept {
    if (op == ElementwiseOp::INVALID || !result_type.is_valid()) { return nullptr; }
    Type result_types[]{std::move(result_type)};
    auto operation = create(OperationKind::ELEMENTWISE, operands, result_types);
    if (operation != nullptr) { operation->_elementwise_op = op; }
    return operation;
}

Operation *IRBuilder::create_mma(Value *a, Value *b, Value *accumulator) noexcept {
    if (a == nullptr || b == nullptr || accumulator == nullptr) { return nullptr; }
    Value *operands[]{a, b, accumulator};
    Type results[]{accumulator->type()};
    return create(OperationKind::MMA, operands, results);
}

Operation *IRBuilder::create_view_load(Value *view, luisa::span<Value *const> indices) noexcept {
    if (view == nullptr || !view->type().is_view()) { return nullptr; }
    luisa::vector<Value *> operands;
    operands.reserve(indices.size() + 1u);
    operands.emplace_back(view);
    operands.insert(operands.end(), indices.begin(), indices.end());
    Type result_types[]{Type::scalar(view->type().scalar_type())};
    return create(OperationKind::VIEW_LOAD, operands, result_types);
}

Operation *IRBuilder::create_view_store(Value *view, luisa::span<Value *const> indices, Value *value) noexcept {
    if (view == nullptr || !view->type().is_view() || value == nullptr) { return nullptr; }
    luisa::vector<Value *> operands;
    operands.reserve(indices.size() + 2u);
    operands.emplace_back(view);
    operands.insert(operands.end(), indices.begin(), indices.end());
    operands.emplace_back(value);
    return create(OperationKind::VIEW_STORE, operands);
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
