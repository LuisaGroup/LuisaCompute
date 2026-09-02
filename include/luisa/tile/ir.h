#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/dll_export.h>
#include <luisa/core/managed_ilist.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/dimension.h>

namespace luisa::compute::tile {

enum class IRForm : uint8_t {
    CANDIDATE,
    SCHEDULED,
    MACHINE
};

enum class ScalarType : uint8_t {
    INVALID,
    BOOL,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64
};

enum class TypeKind : uint8_t {
    INVALID,
    INDEX,
    SCALAR,
    TILE,
    VIEW,
    MEMORY,
    MEMORY_STATE,
    OPAQUE
};

class LUISA_TILE_API Type final {

private:
    TypeKind _kind{TypeKind::INVALID};
    ScalarType _scalar{ScalarType::INVALID};
    luisa::shared_ptr<const IndexSpace> _space;
    luisa::string _opaque_name;

    explicit Type(TypeKind kind) noexcept : _kind{kind} {}
    Type(TypeKind kind, ScalarType scalar, const IndexSpace &space) noexcept;

public:
    Type() noexcept = default;

    [[nodiscard]] static Type index() noexcept { return Type{TypeKind::INDEX}; }
    [[nodiscard]] static Type scalar(ScalarType scalar) noexcept;
    [[nodiscard]] static Type tile(ScalarType scalar, const IndexSpace &space) noexcept;
    [[nodiscard]] static Type view(ScalarType scalar, const IndexSpace &space) noexcept;
    [[nodiscard]] static Type memory(ScalarType scalar, const IndexSpace &space) noexcept;
    [[nodiscard]] static Type memory_state() noexcept { return Type{TypeKind::MEMORY_STATE}; }
    [[nodiscard]] static Type opaque(luisa::string_view name) noexcept;

    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] auto scalar_type() const noexcept { return _scalar; }
    [[nodiscard]] const IndexSpace *index_space() const noexcept { return _space.get(); }
    [[nodiscard]] luisa::string_view opaque_name() const noexcept { return _opaque_name; }
    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] bool is_tile() const noexcept { return _kind == TypeKind::TILE; }
    [[nodiscard]] bool is_view() const noexcept { return _kind == TypeKind::VIEW; }
    [[nodiscard]] bool is_memory() const noexcept { return _kind == TypeKind::MEMORY; }
    [[nodiscard]] Type tile_value_type() const noexcept;

    friend bool operator==(const Type &lhs, const Type &rhs) noexcept;
};

[[nodiscard]] LUISA_TILE_API bool operator==(const Type &lhs, const Type &rhs) noexcept;

class LUISA_TILE_API Attribute final {

public:
    using Value = luisa::variant<luisa::monostate, bool, int64_t, uint64_t, double, luisa::string>;

private:
    Value _value;

public:
    Attribute() noexcept = default;
    explicit Attribute(bool value) noexcept : _value{value} {}
    explicit Attribute(int64_t value) noexcept : _value{value} {}
    explicit Attribute(uint64_t value) noexcept : _value{value} {}
    explicit Attribute(double value) noexcept : _value{value} {}
    explicit Attribute(luisa::string_view value) noexcept : _value{luisa::string{value.data(), value.size()}} {}

    [[nodiscard]] const Value &value() const noexcept { return _value; }
    [[nodiscard]] bool is_valid() const noexcept { return !luisa::holds_alternative<luisa::monostate>(_value); }
};

enum class OperationKind : uint8_t {
    CUSTOM,
    CONSTANT,
    ELEMENTWISE,
    TILE_MAP,
    MMA,
    VIEW_LOAD,
    VIEW_STORE,
    MEMORY_ALLOC,
    MEMORY_LOAD,
    MEMORY_STORE,
    PARALLEL,
    SERIAL,
    PIPELINE,
    STAGE,
    REDUCE,
    YIELD
};

// Closed, target-independent semantics for elementwise scalar or Tile values.
// Backends must dispatch on this opcode, never on a diagnostic string. CUSTOM
// remains available for genuinely extensible dialect operations.
enum class ElementwiseOp : uint8_t {
    INVALID,
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    NEG,
    MIN,
    MAX,
    CAST,
    SELECT,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_NOT,
    EXP,
    LOG,
    SQRT,
    TANH,
    ABS
};

enum class MemoryEffect : uint8_t {
    NONE,
    READ,
    WRITE,
    ALLOCATE,
    UNKNOWN
};

[[nodiscard]] LUISA_TILE_API luisa::string_view to_string(OperationKind kind) noexcept;
[[nodiscard]] LUISA_TILE_API luisa::string_view to_string(ElementwiseOp op) noexcept;

class Module;
class Function;
class Region;
class Block;
class Operation;
class Value;
class Use;
class UseList;
class IRBuilder;
class SentinelOperation;
class SentinelBlock;
class SentinelFunction;

class LUISA_TILE_API Use final : public luisa::ManagedIntrusiveForwardNode<Use> {

private:
    friend class UseList;
    Operation *_user{nullptr};
    size_t _index{0u};
    Value *_value{nullptr};
    UseList *_list_owner{nullptr};

public:
    Use(Operation *user, size_t index) noexcept;
    Use(Use &&) noexcept = delete;
    Use(const Use &) noexcept = delete;
    Use &operator=(Use &&) noexcept = delete;
    Use &operator=(const Use &) noexcept = delete;

    [[nodiscard]] luisa::ManagedPtr<Use> remove_self() noexcept override;
    [[nodiscard]] auto user() noexcept { return _user; }
    [[nodiscard]] const auto *user() const noexcept { return _user; }
    [[nodiscard]] auto index() const noexcept { return _index; }
    [[nodiscard]] auto value() noexcept { return _value; }
    [[nodiscard]] const auto *value() const noexcept { return _value; }
    void set(Value *value) noexcept;
};

// Operand uses are independently owned by their user and linked into exactly
// one defining Value while the user Operation is part of the IR. This gives
// transformations stable Use identities and O(1) detach/re-attach.
class LUISA_TILE_API UseList final {

private:
    luisa::ManagedIntrusiveForwardList<Use> _nodes;

public:
    UseList() noexcept = default;

    [[nodiscard]] bool empty() const noexcept { return _nodes.empty(); }
    [[nodiscard]] Use *front() noexcept { return _nodes.front(); }
    [[nodiscard]] const Use *front() const noexcept { return _nodes.front(); }
    [[nodiscard]] size_t count_size() const noexcept { return _nodes.count_size(); }
    [[nodiscard]] bool contains(const Use *use) const noexcept {
        return use != nullptr && use->_list_owner == this && use->is_linked();
    }

    [[nodiscard]] auto begin() noexcept { return _nodes.begin(); }
    [[nodiscard]] auto begin() const noexcept { return _nodes.begin(); }
    [[nodiscard]] auto end() const noexcept { return _nodes.end(); }
    [[nodiscard]] auto cbegin() const noexcept { return _nodes.cbegin(); }
    [[nodiscard]] auto cend() const noexcept { return _nodes.cend(); }

    Use *push_front(luisa::ManagedPtr<Use> use) noexcept;
};

class LUISA_TILE_API Value final {

public:
    enum class Origin : uint8_t {
        BLOCK_ARGUMENT,
        OPERATION_RESULT
    };

private:
    friend class Block;
    friend class IRBuilder;
    friend class Operation;
    friend class Use;
    uint64_t _id{~0ull};
    Type _type;
    luisa::string _name;
    Origin _origin{Origin::BLOCK_ARGUMENT};
    Block *_block{nullptr};
    Operation *_operation{nullptr};
    size_t _index{0u};
    UseList _use_list;

    Value(uint64_t id, Type type, Block *block, size_t index) noexcept;
    Value(uint64_t id, Type type, Operation *operation, size_t index) noexcept;

public:
    Value(Value &&) noexcept = delete;
    Value(const Value &) noexcept = delete;
    Value &operator=(Value &&) noexcept = delete;
    Value &operator=(const Value &) noexcept = delete;
    ~Value() noexcept;

    [[nodiscard]] auto id() const noexcept { return _id; }
    [[nodiscard]] const Type &type() const noexcept { return _type; }
    [[nodiscard]] luisa::string_view name() const noexcept { return _name; }
    void set_name(luisa::string_view name) noexcept { _name.assign(name.data(), name.size()); }
    [[nodiscard]] auto origin() const noexcept { return _origin; }
    [[nodiscard]] auto index() const noexcept { return _index; }
    [[nodiscard]] Block *argument_block() noexcept { return _block; }
    [[nodiscard]] const Block *argument_block() const noexcept { return _block; }
    [[nodiscard]] Operation *defining_operation() noexcept { return _operation; }
    [[nodiscard]] const Operation *defining_operation() const noexcept { return _operation; }
    [[nodiscard]] size_t use_count() const noexcept { return _use_list.count_size(); }
    [[nodiscard]] UseList &use_list() noexcept { return _use_list; }
    [[nodiscard]] const UseList &use_list() const noexcept { return _use_list; }
    [[nodiscard]] bool replace_all_uses_with(Value *replacement) noexcept;
};

struct NamedAttribute {
    luisa::string name;
    Attribute value;
};

class LUISA_TILE_API Operation : public luisa::ManagedIntrusiveNode<Operation> {

private:
    friend class Block;
    friend class IRBuilder;
    friend class SentinelOperation;
    uint64_t _id{~0ull};
    Block *_parent{nullptr};
    OperationKind _kind{OperationKind::CUSTOM};
    ElementwiseOp _elementwise_op{ElementwiseOp::INVALID};
    luisa::string _custom_name;
    luisa::vector<luisa::ManagedPtr<Use>> _operands;
    luisa::vector<luisa::unique_ptr<Value>> _results;
    luisa::vector<luisa::unique_ptr<Region>> _regions;
    luisa::vector<NamedAttribute> _attributes;
    luisa::optional<IndexSpace> _domain;
    luisa::optional<luisa::string> _execution_scope_constraint;
    luisa::optional<luisa::string> _resource_class_constraint;

    void _remove_self_from_operand_use_lists() noexcept;
    void _add_self_to_operand_use_lists() noexcept;

public:
    Operation(uint64_t id, Block *parent, OperationKind kind, luisa::string_view custom_name = {}) noexcept;
    Operation(Operation &&) noexcept = delete;
    Operation(const Operation &) noexcept = delete;
    Operation &operator=(Operation &&) noexcept = delete;
    Operation &operator=(const Operation &) noexcept = delete;
    ~Operation() noexcept override;

    [[nodiscard]] luisa::ManagedPtr<Operation> remove_self() noexcept override;
    Operation *insert_before_self(luisa::ManagedPtr<Operation> operation) noexcept override;

    [[nodiscard]] auto id() const noexcept { return _id; }
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] auto elementwise_op() const noexcept { return _elementwise_op; }
    [[nodiscard]] luisa::string_view name() const noexcept;
    [[nodiscard]] auto parent_block() noexcept { return _parent; }
    [[nodiscard]] const auto *parent_block() const noexcept { return _parent; }
    [[nodiscard]] Function *parent_function() noexcept;
    [[nodiscard]] const Function *parent_function() const noexcept;
    [[nodiscard]] MemoryEffect memory_effect() const noexcept;

    void add_operand(Value *value) noexcept;
    void set_operand(size_t index, Value *value) noexcept;
    [[nodiscard]] size_t operand_count() const noexcept { return _operands.size(); }
    [[nodiscard]] Use *operand_use(size_t index) noexcept { return _operands[index].get(); }
    [[nodiscard]] const Use *operand_use(size_t index) const noexcept { return _operands[index].get(); }
    [[nodiscard]] Value *operand(size_t index) noexcept { return _operands[index]->value(); }
    [[nodiscard]] const Value *operand(size_t index) const noexcept { return _operands[index]->value(); }

    [[nodiscard]] Value *add_result(Type type) noexcept;
    [[nodiscard]] size_t result_count() const noexcept { return _results.size(); }
    [[nodiscard]] Value *result(size_t index) noexcept { return _results[index].get(); }
    [[nodiscard]] const Value *result(size_t index) const noexcept { return _results[index].get(); }

    [[nodiscard]] Region *add_region(luisa::string_view label = {}) noexcept;
    [[nodiscard]] size_t region_count() const noexcept { return _regions.size(); }
    [[nodiscard]] Region *region(size_t index) noexcept { return _regions[index].get(); }
    [[nodiscard]] const Region *region(size_t index) const noexcept { return _regions[index].get(); }
    [[nodiscard]] const auto &regions() const noexcept { return _regions; }

    void set_domain(IndexSpace domain) noexcept { _domain = std::move(domain); }
    [[nodiscard]] const luisa::optional<IndexSpace> &domain() const noexcept { return _domain; }
    void set_execution_scope_constraint(luisa::string_view scope) noexcept;
    void set_resource_class_constraint(luisa::string_view resource) noexcept;
    [[nodiscard]] const luisa::optional<luisa::string> &execution_scope_constraint() const noexcept { return _execution_scope_constraint; }
    [[nodiscard]] const luisa::optional<luisa::string> &resource_class_constraint() const noexcept { return _resource_class_constraint; }

    void set_attribute(luisa::string_view name, Attribute value) noexcept;
    [[nodiscard]] const Attribute *attribute(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::span<const NamedAttribute> attributes() const noexcept { return _attributes; }
};

class LUISA_TILE_API SentinelOperation final : public Operation {
public:
    explicit SentinelOperation(Block *parent) noexcept;
};

using OperationList = luisa::ManagedIntrusiveList<Operation, SentinelOperation>;

class LUISA_TILE_API Block : public luisa::ManagedIntrusiveNode<Block> {

private:
    friend class Region;
    friend class SentinelBlock;
    Region *_parent{nullptr};
    luisa::vector<luisa::unique_ptr<Value>> _arguments;
    OperationList _operations;

public:
    explicit Block(Region *parent) noexcept;
    Block(Block &&) noexcept = delete;
    Block(const Block &) noexcept = delete;
    Block &operator=(Block &&) noexcept = delete;
    Block &operator=(const Block &) noexcept = delete;
    ~Block() noexcept override;

    [[nodiscard]] luisa::ManagedPtr<Block> remove_self() noexcept override;
    Block *insert_before_self(luisa::ManagedPtr<Block> block) noexcept override;

    [[nodiscard]] auto parent_region() noexcept { return _parent; }
    [[nodiscard]] const auto *parent_region() const noexcept { return _parent; }
    [[nodiscard]] Function *parent_function() noexcept;
    [[nodiscard]] const Function *parent_function() const noexcept;

    [[nodiscard]] Value *add_argument(Type type, luisa::string_view name = {}) noexcept;
    [[nodiscard]] size_t argument_count() const noexcept { return _arguments.size(); }
    [[nodiscard]] Value *argument(size_t index) noexcept { return _arguments[index].get(); }
    [[nodiscard]] const Value *argument(size_t index) const noexcept { return _arguments[index].get(); }
    [[nodiscard]] const auto &arguments() const noexcept { return _arguments; }

    [[nodiscard]] Operation *append_operation(OperationKind kind, luisa::string_view custom_name = {}) noexcept;
    [[nodiscard]] Operation *insert_operation_before(Operation *position,
                                                     OperationKind kind,
                                                     luisa::string_view custom_name = {}) noexcept;
    [[nodiscard]] bool erase(Operation *operation) noexcept;
    [[nodiscard]] size_t operation_count() const noexcept { return _operations.count_size(); }
    [[nodiscard]] Operation *operation(size_t index) noexcept;
    [[nodiscard]] const Operation *operation(size_t index) const noexcept;
    [[nodiscard]] const auto &operations() const noexcept { return _operations; }
    [[nodiscard]] auto &operations() noexcept { return _operations; }
};

class LUISA_TILE_API SentinelBlock final : public Block {
public:
    explicit SentinelBlock(Region *parent) noexcept;
};

using BlockList = luisa::ManagedIntrusiveList<Block, SentinelBlock>;

class LUISA_TILE_API Region final {

private:
    friend class Function;
    friend class Operation;
    Function *_function{nullptr};
    Operation *_parent{nullptr};
    luisa::string _label;
    BlockList _blocks;

    Region(Function *function, Operation *parent, luisa::string_view label) noexcept;

public:
    Region(Region &&) noexcept = delete;
    Region(const Region &) noexcept = delete;
    Region &operator=(Region &&) noexcept = delete;
    Region &operator=(const Region &) noexcept = delete;
    ~Region() noexcept;

    [[nodiscard]] auto parent_operation() noexcept { return _parent; }
    [[nodiscard]] const auto *parent_operation() const noexcept { return _parent; }
    [[nodiscard]] auto parent_function() noexcept { return _function; }
    [[nodiscard]] const auto *parent_function() const noexcept { return _function; }
    [[nodiscard]] luisa::string_view label() const noexcept { return _label; }

    [[nodiscard]] Block *append_block() noexcept;
    [[nodiscard]] size_t block_count() const noexcept { return _blocks.count_size(); }
    [[nodiscard]] Block *block(size_t index) noexcept;
    [[nodiscard]] const Block *block(size_t index) const noexcept;
    [[nodiscard]] const auto &blocks() const noexcept { return _blocks; }
    [[nodiscard]] auto &blocks() noexcept { return _blocks; }
};

class LUISA_TILE_API Function : public luisa::ManagedIntrusiveNode<Function> {

private:
    friend class Block;
    friend class Module;
    friend class Operation;
    friend class SentinelFunction;
    Module *_parent{nullptr};
    uint64_t _id{~0ull};
    luisa::string _name;
    IRForm _form{IRForm::CANDIDATE};
    uint64_t _next_value_id{0u};
    uint64_t _next_operation_id{0u};
    Region _body;

    [[nodiscard]] uint64_t _allocate_value_id() noexcept { return _next_value_id++; }
    [[nodiscard]] uint64_t _allocate_operation_id() noexcept { return _next_operation_id++; }

public:
    Function(Module *parent, uint64_t id, luisa::string_view name, IRForm form) noexcept;
    Function(Function &&) noexcept = delete;
    Function(const Function &) noexcept = delete;
    Function &operator=(Function &&) noexcept = delete;
    Function &operator=(const Function &) noexcept = delete;
    ~Function() noexcept override;

    [[nodiscard]] luisa::ManagedPtr<Function> remove_self() noexcept override;
    Function *insert_before_self(luisa::ManagedPtr<Function> function) noexcept override;

    [[nodiscard]] auto id() const noexcept { return _id; }
    [[nodiscard]] luisa::string_view name() const noexcept { return _name; }
    [[nodiscard]] auto form() const noexcept { return _form; }
    void set_form(IRForm form) noexcept { _form = form; }
    [[nodiscard]] auto parent_module() noexcept { return _parent; }
    [[nodiscard]] const auto *parent_module() const noexcept { return _parent; }
    [[nodiscard]] Region &body() noexcept { return _body; }
    [[nodiscard]] const Region &body() const noexcept { return _body; }
};

class LUISA_TILE_API SentinelFunction final : public Function {
public:
    explicit SentinelFunction(Module *parent) noexcept;
};

using FunctionList = luisa::ManagedIntrusiveList<Function, SentinelFunction>;

class LUISA_TILE_API Module final {

private:
    DimensionContext _dimensions;
    uint64_t _next_function_id{0u};
    FunctionList _functions;

public:
    Module() noexcept;
    Module(Module &&) noexcept = delete;
    Module(const Module &) noexcept = delete;
    Module &operator=(Module &&) noexcept = delete;
    Module &operator=(const Module &) noexcept = delete;
    ~Module() noexcept;

    [[nodiscard]] DimensionContext &dimensions() noexcept { return _dimensions; }
    [[nodiscard]] const DimensionContext &dimensions() const noexcept { return _dimensions; }
    [[nodiscard]] Function *create_function(luisa::string_view name, IRForm form = IRForm::CANDIDATE) noexcept;
    [[nodiscard]] size_t function_count() const noexcept { return _functions.count_size(); }
    [[nodiscard]] Function *function(size_t index) noexcept;
    [[nodiscard]] const Function *function(size_t index) const noexcept;
    [[nodiscard]] const auto &functions() const noexcept { return _functions; }
    [[nodiscard]] auto &functions() noexcept { return _functions; }
};

class LUISA_TILE_API IRBuilder final {

private:
    Block *_insertion_block{nullptr};
    Operation *_insertion_before{nullptr};

public:
    explicit IRBuilder(Block *block = nullptr) noexcept : _insertion_block{block} {}

    void set_insertion_block(Block *block) noexcept {
        _insertion_block = block;
        _insertion_before = nullptr;
    }
    void set_insertion_point(Operation *before) noexcept {
        _insertion_before = before;
        _insertion_block = before == nullptr ? nullptr : before->parent_block();
    }
    [[nodiscard]] auto insertion_block() noexcept { return _insertion_block; }
    [[nodiscard]] const auto *insertion_block() const noexcept { return _insertion_block; }
    [[nodiscard]] auto insertion_before() noexcept { return _insertion_before; }
    [[nodiscard]] const auto *insertion_before() const noexcept { return _insertion_before; }

    [[nodiscard]] Operation *create(OperationKind kind,
                                    luisa::span<Value *const> operands = {},
                                    luisa::span<const Type> result_types = {},
                                    luisa::string_view custom_name = {}) noexcept;
    [[nodiscard]] Operation *create_structured(OperationKind kind,
                                               IndexSpace domain,
                                               luisa::span<Value *const> operands = {},
                                               luisa::span<const Type> result_types = {}) noexcept;
    [[nodiscard]] Operation *create_elementwise(ElementwiseOp op,
                                                luisa::span<Value *const> operands,
                                                Type result_type) noexcept;
    [[nodiscard]] Operation *create_mma(Value *a, Value *b, Value *accumulator) noexcept;
    [[nodiscard]] Operation *create_view_load(Value *view,
                                              luisa::span<Value *const> indices,
                                              Value *predicate = nullptr,
                                              Value *fallback = nullptr) noexcept;
    [[nodiscard]] Operation *create_view_store(Value *view, luisa::span<Value *const> indices, Value *value) noexcept;
    [[nodiscard]] Operation *create_memory_alloc(const Type &memory_type, luisa::string_view resource_class = {}) noexcept;
    [[nodiscard]] Operation *create_memory_load(Value *memory, Value *state) noexcept;
    [[nodiscard]] Operation *create_memory_store(Value *memory, Value *state, Value *tile) noexcept;
};

}// namespace luisa::compute::tile
