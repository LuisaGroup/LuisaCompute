#include <luisa/core/logging.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute::detail {

luisa::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local luisa::vector<FunctionBuilder *> function_builder_stack;
    return function_builder_stack;
}

void FunctionBuilder::push(FunctionBuilder *func) noexcept {
    _function_stack().emplace_back(func);
}

void FunctionBuilder::pop(FunctionBuilder *func) noexcept {
    if (_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid pop on empty function stack.");
    }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    if (func != nullptr && f != func) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid function on stack top.");
    }
    if (f->tag() == Tag::KERNEL &&
        f->_return_type.value_or(nullptr) != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Kernels cannot have non-void return types.");
    }
    if (f->_arguments.size() != f->_bound_arguments.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Arguments and their bindings have different sizes ({} and {}).",
            f->_arguments.size(), f->_bound_arguments.size());
    }
    // non-callables may not have internalizer arguments
    if (f->tag() != Function::Tag::CALLABLE) {
        LUISA_ASSERT(
            f->_captured_external_variables.empty() &&
                f->_internalizer_arguments.empty(),
            "Non-callables may not have internalizer arguments.");
    }

    // hash
    f->_compute_hash();
    if (f->_tag == Function::Tag::KERNEL) {
        f->sort_bindings();
    }

    // clear temporary data
    for (auto p : f->_temporary_data) {
        luisa::detail::allocator_deallocate(
            p.first, p.second);
    }
    f->_temporary_data.clear();
    f->_temporary_data.shrink_to_fit();
}

FunctionBuilder *FunctionBuilder::current() noexcept {
    LUISA_ASSERT(!_function_stack().empty(), "Empty function stack.");
    return _function_stack().back();
}

FunctionBuilder *FunctionBuilder::current_or_null() noexcept {
    return _function_stack().empty() ?
               nullptr :
               _function_stack().back();
}

luisa::span<const FunctionBuilder *const> FunctionBuilder::stack() noexcept {
    return _function_stack();
}

void FunctionBuilder::_append(const Statement *statement) noexcept {
    if (_scope_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Scope stack is empty.");
    }
    _scope_stack.back()->append(statement);
}

void FunctionBuilder::break_() noexcept {
    _create_and_append_statement<BreakStmt>();
}

void FunctionBuilder::continue_() noexcept {
    _create_and_append_statement<ContinueStmt>();
}

void FunctionBuilder::return_(const Expression *expr) noexcept {
    expr = _internalize(expr);
    LUISA_ASSERT(_tag != Tag::KERNEL || expr == nullptr,
                 "Kernels cannot return non-void values.");
    if (_return_type) {// multiple return statements, check if they match
        if (*_return_type == nullptr) {
            LUISA_ASSERT(expr != nullptr,
                         "Mismatched return types: {} and previously void.",
                         expr->type()->description());
        } else {
            LUISA_ASSERT(expr != nullptr && *_return_type.value() == *expr->type(),
                         "Mismatched return types: {} and previously {}.",
                         expr == nullptr ? "void" : expr->type()->description(),
                         _return_type.value()->description());
        }
    }
    if (expr == nullptr) {// returning void
        static thread_local ReturnStmt null_return{nullptr};
        _append(&null_return);
    } else {// returning a non-void value
        _return_type.emplace(expr->type());
        _create_and_append_statement<ReturnStmt>(expr);
    }
}

void FunctionBuilder::suspend_() noexcept {
    suspend_(_next_suspend_token(), luisa::string{}, {});
}

void FunctionBuilder::suspend_(uint32_t token) noexcept {
    suspend_(token, luisa::string{}, {});
}

void FunctionBuilder::suspend_(luisa::string name) noexcept {
    suspend_(_next_suspend_token(), std::move(name), {});
}

void FunctionBuilder::suspend_(uint32_t token, luisa::string name) noexcept {
    suspend_(token, std::move(name), {});
}

void FunctionBuilder::suspend_(
    luisa::string name,
    luisa::vector<CoroFrameExport> frame_exports) noexcept {
    suspend_(_next_suspend_token(), std::move(name),
             std::move(frame_exports));
}

void FunctionBuilder::suspend_(
    uint32_t token, luisa::string name,
    luisa::vector<CoroFrameExport> frame_exports) noexcept {
    LUISA_ASSERT(_tag == Tag::COROUTINE,
                 "Coroutine suspension is only valid in a coroutine.");
    LUISA_ASSERT(token != 0u, "Coroutine suspend token 0 is reserved for coroutine entry.");
    luisa::unordered_set<luisa::string> names;
    for (auto &&frame_export : frame_exports) {
        frame_export.value = _internalize(frame_export.value);
        LUISA_ASSERT(
            frame_export.value != nullptr &&
                frame_export.value->type() != nullptr &&
                frame_export.value->type()->is_basic(),
            "Coroutine frame export '{}' at suspend '{}' must be a "
            "non-null basic value.",
            frame_export.name, name);
        LUISA_ASSERT(!frame_export.name.empty(),
                     "Coroutine frame export names must be non-empty.");
        LUISA_ASSERT(
            names.emplace(frame_export.name).second,
            "Duplicate coroutine frame export '{}' at suspend '{}'.",
            frame_export.name, name);
    }
    _create_and_append_statement<SuspendStmt>(
        token, std::move(name), std::move(frame_exports));
}

RayQueryStmt *FunctionBuilder::ray_query_(const RefExpr *query) noexcept {
    LUISA_ASSERT(query->builder() == this,
                 "Ray query must be created by the same function builder.");
    return _create_and_append_statement<RayQueryStmt>(query);
}

AutoDiffStmt *FunctionBuilder::autodiff_() noexcept {
    return _create_and_append_statement<AutoDiffStmt>();
}

IfStmt *FunctionBuilder::if_(const Expression *cond) noexcept {
    cond = _internalize(cond);
    return _create_and_append_statement<IfStmt>(cond);
}

LoopStmt *FunctionBuilder::loop_() noexcept {
    return _create_and_append_statement<LoopStmt>();
}

void FunctionBuilder::mark_loop_as_while(
    LoopStmt *loop, const Expression *condition,
    size_t condition_statement_count) noexcept {
    LUISA_ASSERT(loop != nullptr && condition != nullptr,
                 "A DSL while loop and its condition must not be null.");
    condition = _internalize(condition);
    LUISA_ASSERT(loop->_while_condition == nullptr,
                 "A DSL loop was marked as while more than once.");
    LUISA_ASSERT(condition_statement_count <=
                     loop->body()->statements().size(),
                 "DSL while condition statement count is out of range.");
    loop->_while_condition = condition;
    loop->_while_condition_statement_count =
        condition_statement_count;
}

void FunctionBuilder::_void_expr(const Expression *expr) noexcept {
    expr = _internalize(expr);
    if (expr != nullptr) { _create_and_append_statement<ExprStmt>(expr); }
}

SwitchStmt *FunctionBuilder::switch_(const Expression *expr) noexcept {
    expr = _internalize(expr);
    return _create_and_append_statement<SwitchStmt>(expr);
}

SwitchCaseStmt *FunctionBuilder::case_(const Expression *expr) noexcept {
    expr = _internalize(expr);
    return _create_and_append_statement<SwitchCaseStmt>(expr);
}

SwitchDefaultStmt *FunctionBuilder::default_() noexcept {
    return _create_and_append_statement<SwitchDefaultStmt>();
}

void FunctionBuilder::assign(const Expression *lhs, const Expression *rhs) noexcept {
    lhs = _internalize(lhs);
    rhs = _internalize(rhs);
    if (lhs->tag() == Expression::Tag::MEMBER) [[unlikely]] {
        auto mem_expr = static_cast<MemberExpr const *>(lhs);
        if (mem_expr->is_swizzle() && mem_expr->swizzle_size() > 1) [[unlikely]] {
            // switch (self->tag()) {
            //     case Expression::Tag::MEMBER: {
            //         if (static_cast<MemberExpr const *>(self)->is_swizzle()) {
            //             LUISA_ERROR("Can not use multiple swizzle write.");
            //         }
            //     } break;
            //     case Expression::Tag::ACCESS: break;
            //     case Expression::Tag::REF: break;
            //     default:
            //         LUISA_ERROR("Invalid swizzle");
            //         break;
            // }
            auto non_trivial = [&](Expression const *expr) -> bool {
                switch (expr->tag()) {
                    case Expression::Tag::LITERAL:
                    case Expression::Tag::REF:
                        return false;
                    default:
                        return true;
                }
            };
            auto temp_var = [&](Expression const *expr) {
                auto local_var = local(expr->type());
                assign(local_var, expr);
                return local_var;
            };
            auto access_chain_decode = [&](auto &&self, Expression const *expr) -> Expression const * {
                switch (expr->tag()) {
                    case Expression::Tag::MEMBER: {
                        auto mem = static_cast<MemberExpr const *>(expr);
                        if (mem->is_swizzle()) [[unlikely]] {
                            LUISA_ERROR("Can not use multiple swizzle write.");
                        }
                        auto mem_self = mem->self();
                        auto new_self = self(self, mem_self);
                        if (new_self != mem_self) {
                            return member(new_self->type(), new_self, mem->member_index());
                        }
                        return expr;
                    }
                    case Expression::Tag::ACCESS: {
                        auto access_expr = static_cast<AccessExpr const *>(expr);
                        auto new_range = self(self, access_expr->range());
                        if (non_trivial(access_expr->index())) {
                            return access(access_expr->type(), new_range, temp_var(access_expr->index()));
                        }
                        if (new_range != access_expr->range()) {
                            return access(access_expr->type(), new_range, access_expr->index());
                        }
                        return access_expr;
                    }
                    case Expression::Tag::REF: return expr;
                    default: LUISA_ERROR("Invalid swizzle");
                }
                return nullptr;
            };
            auto self = mem_expr->self();
            self = access_chain_decode(access_chain_decode, self);
            auto local_var = local(mem_expr->type());
            auto elem = mem_expr->type()->element();
            _create_and_append_statement<AssignStmt>(local_var, rhs);
            std::array<const Expression *, 4> exprs{};
            for (auto i = 0; i < mem_expr->swizzle_size(); ++i) {
                exprs[mem_expr->swizzle_index(i)] = swizzle(elem, local_var, 1, i);
            }
            auto make_type = self->type();
            for (auto i = 0; i < make_type->dimension(); ++i) {
                if (exprs[i] == nullptr) {
                    exprs[i] = swizzle(elem, self, 1, i);
                }
            }
            auto call_expr = make_vector(make_type, {exprs.data(), make_type->dimension()});
            _create_and_append_statement<AssignStmt>(self, call_expr);
            return;
        }
    }
    _create_and_append_statement<AssignStmt>(lhs, rhs);
}

const CallExpr *FunctionBuilder::make_vector(const Type *type, luisa::span<const Expression *const> args) noexcept {
    LUISA_ASSERT(type->tag() == Type::Tag::VECTOR, "Must be vector type.");
    auto elem = type->element();
    auto op = static_cast<CallOp>(
        luisa::to_underlying([&]() {
            switch (elem->tag()) {
                case Type::Tag::BOOL:
                    return CallOp::MAKE_BOOL2;
                case Type::Tag::UINT16:
                    return CallOp::MAKE_USHORT2;
                case Type::Tag::UINT32:
                    return CallOp::MAKE_UINT2;
                case Type::Tag::UINT64:
                    return CallOp::MAKE_ULONG2;
                case Type::Tag::INT16:
                    return CallOp::MAKE_SHORT2;
                case Type::Tag::INT32:
                    return CallOp::MAKE_INT2;
                case Type::Tag::INT64:
                    return CallOp::MAKE_LONG2;
                case Type::Tag::FLOAT16:
                    return CallOp::MAKE_HALF2;
                case Type::Tag::FLOAT32:
                    return CallOp::MAKE_FLOAT2;
                case Type::Tag::FLOAT64:
                    return CallOp::MAKE_DOUBLE2;
                default:
                    LUISA_ERROR("Invalid element type.");
                    return CallOp::MAKE_BOOL2;
            }
        }()) +
        elem->dimension() - 2);
    return call(type, op, args);
}

const LiteralExpr *FunctionBuilder::literal(const Type *type, LiteralExpr::Value value) noexcept {
    luisa::visit(
        [type]<typename T>(T x) noexcept {
            auto t = Type::of<T>();
            LUISA_ASSERT(*type == *t,
                         "Type mismatch: declared as {}, got {}.",
                         type->description(), t->description());
        },
        value);
    return _create_expression<LiteralExpr>(type, value);
}

const RefExpr *FunctionBuilder::local(const Type *type) noexcept {
    _use_cooperative_operations |= type->is_cooperative_vector();
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    _local_variables.emplace_back(v);
    return _ref(v);
}
void FunctionBuilder::set_variable_name(uint32_t id, luisa::string_view name) const noexcept {
    if (_variables_names.size() <= id) {
        _variables_names.resize(id + 1);
    }
    _variables_names[id] = luisa::string{name};
}

luisa::string_view FunctionBuilder::get_variable_name(uint32_t uid) const noexcept {
    if (uid < _variables_names.size()) {
        return _variables_names[uid];
    }
    return {};
}

const RefExpr *FunctionBuilder::shared(const Type *type) noexcept {
    if (type->is_structure() && !type->member_attributes().empty()) [[unlikely]] {
        LUISA_ERROR("Shared variable can not be structure type with custom attributes");
    }
    Variable sv{type, Variable::Tag::SHARED, _next_variable_uid()};
    _shared_variables.emplace_back(sv);
    return _ref(sv);
}

uint32_t FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}

uint32_t FunctionBuilder::_next_suspend_token() noexcept {
    return _next_coro_suspend_token++;
}

const RefExpr *FunctionBuilder::thread_id() noexcept { return _builtin(Type::of<uint3>(), Variable::Tag::THREAD_ID); }
const RefExpr *FunctionBuilder::block_id() noexcept { return _builtin(Type::of<uint3>(), Variable::Tag::BLOCK_ID); }
const RefExpr *FunctionBuilder::dispatch_id() noexcept { return _builtin(Type::of<uint3>(), Variable::Tag::DISPATCH_ID); }
const RefExpr *FunctionBuilder::dispatch_size() noexcept { return _builtin(Type::of<uint3>(), Variable::Tag::DISPATCH_SIZE); }
const RefExpr *FunctionBuilder::kernel_id() noexcept { return _builtin(Type::of<uint>(), Variable::Tag::KERNEL_ID); }
const RefExpr *FunctionBuilder::raster_object_id() noexcept { return _builtin(Type::of<uint>(), Variable::Tag::RASTER_OBJECT_ID); }
const RefExpr *FunctionBuilder::raster_barycentrics() noexcept { return _builtin(Type::of<uint>(), Variable::Tag::RASTER_BARYCENTRICS); }
const RefExpr *FunctionBuilder::warp_lane_count() noexcept { return _builtin(Type::of<uint>(), Variable::Tag::WARP_LANE_COUNT); }
const RefExpr *FunctionBuilder::warp_lane_id() noexcept { return _builtin(Type::of<uint>(), Variable::Tag::WARP_LANE_ID); }

const RefExpr *FunctionBuilder::coro_token() noexcept { return local(Type::of<uint>()); }

inline const RefExpr *FunctionBuilder::_builtin(Type const *type, Variable::Tag tag) noexcept {
    if (auto iter = std::find_if(
            _builtin_variables.cbegin(), _builtin_variables.cend(),
            [tag](auto &&v) noexcept { return v.tag() == tag; });
        iter != _builtin_variables.cend()) {
        return _ref(*iter);
    }
    Variable v{type, tag, _next_variable_uid()};
    _builtin_variables.emplace_back(v);
    // for callables, builtin variables are treated like arguments
    if (_tag == Function::Tag::CALLABLE) [[unlikely]] {
        _arguments.emplace_back(v);
        _bound_arguments.emplace_back();
    }
    return _ref(v);
}

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes, size_t size_bytes) noexcept {
    // find if already bound
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        return *_arguments[i].type() == *type &&
                               binding.handle == handle &&
                               binding.offset == offset_bytes;
                    } else {
                        return false;
                    }
                },
                _bound_arguments[i])) {
            auto &binding = luisa::get<Function::BufferBinding>(_bound_arguments[i]);
            binding.size = std::max(binding.size, size_bytes);
            return _ref(_arguments[i]);
        }
    }
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back(Function::BufferBinding{handle, offset_bytes, size_bytes});
    return _ref(v);
}

const UnaryExpr *FunctionBuilder::unary(const Type *type, UnaryOp op, const Expression *expr) noexcept {
    expr = _internalize(expr);
    return _create_expression<UnaryExpr>(type, op, expr);
}

const BinaryExpr *FunctionBuilder::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    lhs = _internalize(lhs);
    rhs = _internalize(rhs);
    return _create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const MemberExpr *FunctionBuilder::member(const Type *type, const Expression *self, size_t member_index) noexcept {
    self = _internalize(self);
    return _create_expression<MemberExpr>(type, self, member_index);
}

const Expression *FunctionBuilder::swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept {
    self = _internalize(self);
    // special handling for literal swizzles
    if (self->tag() == Expression::Tag::LITERAL) {
        auto element = luisa::visit(
            [&](auto &&v) noexcept -> const Expression * {
                using TVec = std::decay_t<decltype(v)>;
                if constexpr (is_vector_v<TVec>) {
                    using TElem = vector_element_t<TVec>;
                    if (swizzle_size == 1u) {
                        auto i = swizzle_code & 0b11u;
                        return literal(Type::of<TElem>(), v[i]);
                    } else if (swizzle_size == 2u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        return literal(Type::of<Vector<TElem, 2u>>(),
                                       Vector<TElem, 2u>{v[i], v[j]});
                    } else if (swizzle_size == 3u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        auto k = (swizzle_code >> 8u) & 0b11u;
                        return literal(Type::of<Vector<TElem, 3u>>(),
                                       Vector<TElem, 3u>{v[i], v[j], v[k]});
                    } else if (swizzle_size == 4u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        auto k = (swizzle_code >> 8u) & 0b11u;
                        auto l = (swizzle_code >> 12u) & 0b11u;
                        return literal(Type::of<Vector<TElem, 4u>>(),
                                       Vector<TElem, 4u>{v[i], v[j], v[k], v[l]});
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid swizzle size.");
                } else {
                    LUISA_ERROR_WITH_LOCATION("Swizzle must be a vector but got '{}'.",
                                              self->type()->description());
                }
            },
            static_cast<const LiteralExpr *>(self)->value());
        return element;
    }
    return _create_expression<MemberExpr>(type, self, swizzle_size, swizzle_code);
}

const AccessExpr *FunctionBuilder::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    range = _internalize(range);
    index = _internalize(index);
    if (range->tag() == Expression::Tag::LITERAL) {
        auto v = local(range->type());
        assign(v, range);
        range = v;
    }
    return _create_expression<AccessExpr>(type, range, index);
}

const CastExpr *FunctionBuilder::cast(const Type *type, CastOp op, const Expression *expr) noexcept {
    expr = _internalize(expr);
    return _create_expression<CastExpr>(type, op, expr);
}

const StringIDExpr *FunctionBuilder::string_id(luisa::string s) noexcept {
    return _create_expression<StringIDExpr>(std::move(s));
}

const TypeIDExpr *FunctionBuilder::type_id(const Type *payload) noexcept {
    return _create_expression<TypeIDExpr>(payload);
}

const RefExpr *FunctionBuilder::_ref(Variable v) noexcept {
    return _create_expression<RefExpr>(v);
}

const ConstantExpr *FunctionBuilder::constant(const ConstantData &c) noexcept {
    if (auto iter = std::find_if(
            _captured_constants.begin(), _captured_constants.end(),
            [c](auto &&cc) noexcept { return c.hash() == cc.hash(); });
        iter == _captured_constants.end()) {
        _captured_constants.emplace_back(c);
    }
    return _create_expression<ConstantExpr>(c);
}

const Statement *FunctionBuilder::pop_stmt() noexcept {
    return _scope_stack.back()->pop();
}

void FunctionBuilder::push_scope(ScopeStmt *s) noexcept {
    _scope_stack.emplace_back(s);
}

void FunctionBuilder::pop_scope(const ScopeStmt *s) noexcept {
    if (_scope_stack.empty() ||
        (s != nullptr && _scope_stack.back() != s)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid scope stack pop.");
    }
    _scope_stack.pop_back();
}

bool FunctionBuilder::inside_function_scope() const noexcept {
    return !_scope_stack.empty() && _scope_stack.back() == &_body;
}

ForStmt *FunctionBuilder::for_(const Expression *var, const Expression *condition, const Expression *update) noexcept {
    var = _internalize(var);
    condition = _internalize(condition);
    update = _internalize(update);
    return _create_and_append_statement<ForStmt>(var, condition, update);
}

void FunctionBuilder::mark_variable_usage(uint32_t uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}

FunctionBuilder::~FunctionBuilder() noexcept = default;

FunctionBuilder::FunctionBuilder(FunctionBuilder::Tag tag) noexcept
    : _hash{0ul}, _tag{tag} {}

const RefExpr *FunctionBuilder::texture(const Type *type) noexcept {
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::texture_binding(const Type *type, uint64_t handle, uint32_t level) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        return *_arguments[i].type() == *type &&
                               binding.handle == handle &&
                               binding.level == level;
                    } else {
                        return false;
                    }
                },
                _bound_arguments[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{
        type,
        Variable::Tag::TEXTURE,
        _next_variable_uid(),
    };
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back(Function::TextureBinding{handle, level});
    return _ref(v);
}

const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, std::initializer_list<const Expression *> args, CurveBasisSet curve_basis_set) noexcept {
    luisa::vector<const Expression *> arg_list{args};
    return call(type, call_op, arg_list, curve_basis_set);
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, std::initializer_list<const Expression *> args) noexcept {
    luisa::vector<const Expression *> arg_list{args};
    return call(type, custom, arg_list);
}

void FunctionBuilder::call(CallOp call_op, std::initializer_list<const Expression *> args) noexcept {
    static_cast<void>(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, std::initializer_list<const Expression *> args) noexcept {
    static_cast<void>(call(nullptr, custom, args));
}

bool FunctionBuilder::operator==(const FunctionBuilder &rhs) const noexcept {
    // Preserve builder identity while calls are being constructed. Two
    // structurally equivalent builders may carry different resource bindings;
    // FunctionBuilder::call() must materialize each builder's bindings before
    // the completed call graph can be canonicalized safely by hash.
    return std::addressof(rhs) == this;
}

void FunctionBuilder::_compute_hash() noexcept {
    if (_hash_computed) {
        LUISA_WARNING_WITH_LOCATION("Hash already computed.");
    }
    using namespace std::string_view_literals;
    // This is a structural hash of everything that can affect generated code
    // or its ABI. Keep field groups explicitly delimited: concatenating the
    // vectors alone makes different field partitions indistinguishable.
    // Resource binding payloads are intentionally absent. They are runtime
    // call-site data; only the binding kind participates in the function ABI.
    // Names and function attributes are debug/user metadata and are not read
    // by code generation.
    static auto function_builder_seed = hash_value("__hash_function_v2"sv);
    enum struct Field : uint64_t {
        tag,
        body,
        return_type,
        arguments,
        argument_binding_kinds,
        builtin_variables,
        local_variables,
        shared_variables,
        variable_usages,
        captured_constants,
        required_curve_bases,
        direct_builtin_callables,
        propagated_builtin_callables,
        launch_config,
        semantic_flags,
    };
    luisa::vector<uint64_t> hashes;
    hashes.reserve(32u +
                   _arguments.size() +
                   _bound_arguments.size() +
                   _builtin_variables.size() +
                   _local_variables.size() +
                   _shared_variables.size() +
                   _captured_constants.size() +
                   _variable_usages.size());
    auto begin_group = [&hashes](Field field, size_t count) noexcept {
        hashes.emplace_back(static_cast<uint64_t>(field));
        hashes.emplace_back(static_cast<uint64_t>(count));
    };

    begin_group(Field::tag, 1u);
    hashes.emplace_back(hash_value(_tag));

    begin_group(Field::body, 2u);
    hashes.emplace_back(_body.hash());
    hashes.emplace_back(_body.statements().size());

    begin_group(Field::return_type, 1u);
    hashes.emplace_back(_return_type ? hash_value(*_return_type.value()) : hash_value("void"sv));

    begin_group(Field::arguments, _arguments.size());
    for (auto &&arg : _arguments) { hashes.emplace_back(arg.hash()); }

    begin_group(Field::argument_binding_kinds, _bound_arguments.size());
    for (auto &&binding : _bound_arguments) {
        hashes.emplace_back(static_cast<uint64_t>(binding.index()));
    }

    begin_group(Field::builtin_variables, _builtin_variables.size());
    for (auto &&variable : _builtin_variables) { hashes.emplace_back(variable.hash()); }

    begin_group(Field::local_variables, _local_variables.size());
    for (auto &&variable : _local_variables) { hashes.emplace_back(variable.hash()); }

    begin_group(Field::shared_variables, _shared_variables.size());
    for (auto &&variable : _shared_variables) { hashes.emplace_back(variable.hash()); }

    begin_group(Field::variable_usages, _variable_usages.size());
    for (auto &&arg : _variable_usages) { hashes.emplace_back(hash_value(arg)); }

    begin_group(Field::captured_constants, _captured_constants.size());
    for (auto &&constant : _captured_constants) {
        hashes.emplace_back(hash_combine({constant.type()->hash(), constant.hash()}));
    }

    begin_group(Field::required_curve_bases, 1u);
    hashes.emplace_back(_required_curve_bases.hash());

    auto append_call_ops = [&begin_group, &hashes](Field field,
                                                   const CallOpSet &ops) noexcept {
        auto count = 0u;
        for ([[maybe_unused]] auto op : ops) { count++; }
        begin_group(field, count);
        for (auto op : ops) { hashes.emplace_back(hash_value(op)); }
    };
    append_call_ops(Field::direct_builtin_callables, _direct_builtin_callables);
    append_call_ops(Field::propagated_builtin_callables, _propagated_builtin_callables);

    begin_group(Field::launch_config, 2u);
    hashes.emplace_back(hash_value(_block_size));
    hashes.emplace_back(_allowed_warp_size);

    begin_group(Field::semantic_flags, 1u);
    hashes.emplace_back(static_cast<uint64_t>(_requires_atomic_float) |
                        static_cast<uint64_t>(_requires_printing) << 1u |
                        static_cast<uint64_t>(_use_cooperative_operations) << 2u);

    _hash = hash64(hashes.data(), hashes.size() * sizeof(uint64_t), function_builder_seed);
    _hash_computed = true;
}

luisa::string FunctionBuilder::debug_name() const noexcept {
    auto tag_str = [this] {
        using namespace std::string_view_literals;
        switch (_tag) {
            case Tag::CALLABLE: return "callable"sv;
            case Tag::KERNEL: return "kernel"sv;
            case Tag::RASTER_STAGE: return "raster_stage"sv;
            case Tag::COROUTINE: return "coroutine"sv;
        }
        return "unknown"sv;
    }();
    return _name.empty() ?
               luisa::format("{}_{:016x}", tag_str, hash()) :
               luisa::format("{}_{}_{:016x}", tag_str, _name, hash());
}

uint64_t FunctionBuilder::hash() const noexcept {
    LUISA_ASSERT(_hash_computed, "Hash not computed.");
    return _hash;
}

const RefExpr *FunctionBuilder::bindless_array_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _bound_arguments[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back(Function::BindlessArrayBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::bindless_array() noexcept {
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _bound_arguments[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back(Function::AccelBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel() noexcept {
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

// call builtin functions
const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, luisa::span<const Expression *const> args, CurveBasisSet curve_basis_set) noexcept {

    if (call_op == CallOp::CUSTOM) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom functions are not allowed to "
            "be called with enum CallOp.");
    }
    _direct_builtin_callables.mark(call_op);
    _propagated_builtin_callables.mark(call_op);
    if (is_atomic_operation(call_op)) {
        LUISA_ASSERT(!args.empty(), "Atomic operation requires at least one argument.");
        if (args.front()->type()->element()->is_float32()) {
            _requires_atomic_float = true;
        }
    }
    CallExpr::ArgumentList internalized_args;
    internalized_args.reserve(args.size());
    for (auto arg : args) { internalized_args.emplace_back(_internalize(arg)); }
    auto expr = _create_expression<CallExpr>(type, call_op, internalized_args, curve_basis_set);
    if (type == nullptr) {
        _void_expr(expr);
        return nullptr;
    }
    ////////// argument check
    return expr;
}

const CallExpr *FunctionBuilder::call(const Type *type,
                                      luisa::shared_ptr<const ExternalFunction> func,
                                      luisa::span<const Expression *const> args) noexcept {
    CallExpr::ArgumentList internalized_args;
    internalized_args.reserve(args.size());
    for (auto arg : args) { internalized_args.emplace_back(_internalize(arg)); }
    auto expr = _create_expression<CallExpr>(type, func.get(), internalized_args);
    if (auto iter = std::find(_used_external_functions.cbegin(),
                              _used_external_functions.cend(), func);
        iter == _used_external_functions.cend()) {
        _used_external_functions.emplace_back(std::move(func));
    }
    if (type == nullptr) {
        _void_expr(expr);
        return nullptr;
    }
    return expr;
}

void FunctionBuilder::mark_required_curve_basis(CurveBasis basis) noexcept {
    _required_curve_bases.mark(basis);
}

void FunctionBuilder::mark_required_curve_basis_set(CurveBasisSet basis_set) noexcept {
    _required_curve_bases.propagate(basis_set);
}

void FunctionBuilder::call(luisa::shared_ptr<const ExternalFunction> func,
                           luisa::span<const Expression *const> args) noexcept {
    static_cast<void>(call(nullptr, std::move(func), args));
}

// call custom functions

const FuncRefExpr *FunctionBuilder::func_ref(Function custom) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION(
            "Calling non-callable function in device code.");
    }
    auto iter = _used_custom_callables.emplace(custom.shared_builder());
    auto f = iter.first->get();
    if (iter.second) {
        // propagate used builtin/custom callables and constants
        _propagated_builtin_callables.propagate(f->_propagated_builtin_callables);
        _required_curve_bases.propagate(f->_required_curve_bases);
        _use_cooperative_operations |= f->_use_cooperative_operations;
        _requires_atomic_float |= f->_requires_atomic_float;
        _requires_printing |= f->_requires_printing;
    }
    auto expr = _create_expression<FuncRefExpr>(f);
    return expr;
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, luisa::span<const Expression *const> args) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION(
            "Calling non-callable function in device code.");
    }
    auto f = custom.builder();
    auto iter = _used_custom_callables.emplace(custom.shared_builder());
    f = iter.first->get();
    CallExpr::ArgumentList call_args(f->_arguments.size(), nullptr);
    auto in_iter = args.begin();
    for (auto i = 0u; i < f->_arguments.size(); i++) {
        if (auto arg = f->_arguments[i]; arg.is_builtin()) {
            call_args[i] = _builtin(arg.type(), arg.tag());
        } else {
            call_args[i] = luisa::visit(
                [&]<typename T>(T binding) noexcept -> const Expression * {
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        return buffer_binding(f->_arguments[i].type(), binding.handle, binding.offset, binding.size);
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        return texture_binding(f->_arguments[i].type(), binding.handle, binding.level);
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        return bindless_array_binding(binding.handle);
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        return accel_binding(binding.handle);
                    } else {
                        // if the argument is captured from the outer scope,
                        // implicitly pass it to the callee with the captured value
                        if (auto iter = f->_internalizer_arguments.find(arg);
                            iter != f->_internalizer_arguments.end()) {
                            return _internalize(iter->second);
                        }
                        // normal argument
                        LUISA_ASSERT(in_iter != args.end(),
                                     "Not enough arguments for custom callable.");
                        return _internalize(*(in_iter++));
                    }
                },
                f->_bound_arguments[i]);
        }
    }
    if (in_iter != args.end()) [[unlikely]] {
        luisa::string expected_args{"("};
        for (auto a : f->_arguments) {
            expected_args.append("\n    ").append(a.type()->description()).append(",");
        }
        if (!f->_arguments.empty()) {
            expected_args.pop_back();
            expected_args.append("\n");
        }
        expected_args.append(")");
        luisa::string received_args{"("};
        for (auto a : args) {
            received_args.append("\n    ").append(a->type()->description()).append(",");
        }
        if (!args.empty()) {
            received_args.pop_back();
            received_args.append("\n");
        }
        received_args.append(")");
        LUISA_ERROR_WITH_LOCATION(
            "Invalid call arguments for custom callable #{:016x}.\n"
            "Expected: {},\n"
            "Received: {}.",
            custom.hash(), expected_args, received_args);
    }

    if (iter.second) {
        // propagate used builtin/custom callables and constants
        _propagated_builtin_callables.propagate(f->_propagated_builtin_callables);
        _required_curve_bases.propagate(f->_required_curve_bases);
        _use_cooperative_operations |= f->_use_cooperative_operations;
        _requires_atomic_float |= f->_requires_atomic_float;
        _requires_printing |= f->_requires_printing;
    }
    auto expr = _create_expression<CallExpr>(type, Function(iter.first->get()), std::move(call_args));
    return expr;
}

const CpuCustomOpExpr *FunctionBuilder::call(const Type *type, void (*f)(void *, void *), void (*dtor)(void *), void *data, const Expression *arg) noexcept {
    auto expr = _create_expression<CpuCustomOpExpr>(type, f, dtor, data, arg);
    return expr;
}

void FunctionBuilder::call(CallOp call_op, luisa::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, luisa::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, custom, args));
}

const RefExpr *FunctionBuilder::reference(const Type *type) noexcept {
    Variable v{type, Variable::Tag::REFERENCE, _next_variable_uid()};
    _arguments.emplace_back(v);
    _bound_arguments.emplace_back();
    return _ref(v);
}

void FunctionBuilder::comment_(luisa::string comment) noexcept {
    _create_and_append_statement<CommentStmt>(std::move(comment));
}

void FunctionBuilder::print_(luisa::string format,
                             luisa::span<const Expression *const> args) noexcept {
    CallExpr::ArgumentList internalized_args;
    internalized_args.reserve(args.size());
    for (auto arg : args) {
        internalized_args.emplace_back(_internalize(arg));
    }
    _create_and_append_statement<PrintStmt>(
        std::move(format),
        std::move(internalized_args));
    _requires_printing = true;
}

void FunctionBuilder::debug_break_(DebugBreakStmt::Wrapper *wrapper,
                                   luisa::span<const Expression *const> watches) noexcept {
    CallExpr::ArgumentList internalized_args;
    internalized_args.reserve(watches.size());
    for (auto &&w : watches) {
        internalized_args.emplace_back(_internalize(w));
    }
    _create_and_append_statement<DebugBreakStmt>(wrapper, std::move(internalized_args));
}

void FunctionBuilder::set_block_size(uint3 size) noexcept {
    if (_tag == Tag::KERNEL) {
        auto kernel_size = size.x * size.y * size.z;
        if (kernel_size == 0 || kernel_size > 1024) [[unlikely]] {
            LUISA_ERROR("Function block size must be in range [1, 1024], Current block size is: {}.",
                        kernel_size);
        }
        if (any(size == uint3(0))) [[unlikely]] {
            LUISA_ERROR("Function block size must be larger than 0, Current block size is: [{}, {}, {}].",
                        size.x, size.y, size.z);
        }
        if (size.z > 64) [[unlikely]] {
            LUISA_ERROR("Function block z-axis's size must be less or equal than 64, Current block size is: {}.",
                        size.z);
        }
        _block_size = size;
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Setting block size is not valid in Callables. "
            "Ignoring the `set_block_size({}, {}, {})` call.",
            size.x, size.y, size.z);
    }
}

void FunctionBuilder::set_name(luisa::string_view name) const noexcept {
    _name = name;
    // canonicalize the name
    for (auto &c : _name) {
        if (!isalnum(c) && c != '_') {
            c = '_';
        }
    }
}

bool FunctionBuilder::requires_raytracing() const noexcept {
    return _propagated_builtin_callables.uses_raytracing();
}

bool FunctionBuilder::requires_motion_blur() const noexcept {
    return _propagated_builtin_callables.uses_raytracing_motion_blur();
}

bool FunctionBuilder::requires_atomic() const noexcept {
    return _propagated_builtin_callables.uses_atomic();
}

bool FunctionBuilder::requires_atomic_float() const noexcept {
    return _requires_atomic_float;
}

bool FunctionBuilder::requires_autodiff() const noexcept {
    return _propagated_builtin_callables.uses_autodiff();
}

bool FunctionBuilder::requires_printing() const noexcept {
    return _requires_printing;
}

void FunctionBuilder::sort_bindings() noexcept {
    luisa::vector<Variable> new_args;
    luisa::vector<Binding> new_bindings;
    new_args.reserve(_arguments.size());
    new_bindings.reserve(_bound_arguments.size());
    // get capture first
    for (size_t i = 0; i < _arguments.size(); ++i) {
        auto &bind = _bound_arguments[i];
        if (!holds_alternative<monostate>(bind)) {
            new_args.emplace_back(_arguments[i]);
            new_bindings.emplace_back(bind);
        }
    }
    for (size_t i = 0; i < _arguments.size(); ++i) {
        auto &bind = _bound_arguments[i];
        if (holds_alternative<monostate>(bind)) {
            new_args.emplace_back(_arguments[i]);
        }
    }
    _arguments = std::move(new_args);
    _bound_arguments = std::move(new_bindings);
}

static void check_expr_is_internalizable(const Expression *expr) noexcept {
    // check if the expression can be internalized
    switch (expr->tag()) {
        case Expression::Tag::MEMBER: {
            auto e = static_cast<const MemberExpr *>(expr);
            check_expr_is_internalizable(e->self());
            break;
        }
        case Expression::Tag::ACCESS: {
            auto range = static_cast<const AccessExpr *>(expr)->range();
            check_expr_is_internalizable(range);
            break;
        }
        case Expression::Tag::REF: {
            auto v = static_cast<const RefExpr *>(expr)->variable();
            LUISA_ASSERT(v.tag() != Variable::Tag::SHARED,
                         "Cannot internalize shared variable.");
            break;
        }
        default: break;
    }
}

[[nodiscard]] static bool expr_is_lvalue_local_variable(const Expression *expr) noexcept {
    switch (expr->tag()) {
        case Expression::Tag::MEMBER: {
            auto e = static_cast<const MemberExpr *>(expr);
            auto is_rvalue_swizzle = e->is_swizzle() && e->swizzle_size() != 1u;
            return !is_rvalue_swizzle && expr_is_lvalue_local_variable(e->self());
        }
        case Expression::Tag::ACCESS: {
            auto range = static_cast<const AccessExpr *>(expr)->range();
            return expr_is_lvalue_local_variable(range);
        }
        case Expression::Tag::REF: {
            auto v = static_cast<const RefExpr *>(expr)->variable();
            return v.tag() == Variable::Tag::LOCAL ||
                   v.tag() == Variable::Tag::REFERENCE;
        }
        default: break;
    }
    return false;
}

bool is_expr_statically_evaluated(const Expression *expr) noexcept {
    switch (expr->tag()) {
        case Expression::Tag::UNARY: {
            return is_expr_statically_evaluated(
                static_cast<UnaryExpr const *>(expr)->operand());
        }
        case Expression::Tag::BINARY: {
            auto e = static_cast<BinaryExpr const *>(expr);
            return is_expr_statically_evaluated(e->lhs()) &&
                   is_expr_statically_evaluated(e->rhs());
        }
        case Expression::Tag::MEMBER: {
            return is_expr_statically_evaluated(
                static_cast<MemberExpr const *>(expr)->self());
        }
        case Expression::Tag::ACCESS: {
            auto e = static_cast<AccessExpr const *>(expr);
            return is_expr_statically_evaluated(e->index()) &&
                   is_expr_statically_evaluated(e->range());
        }
        case Expression::Tag::LITERAL:
        case Expression::Tag::CONSTANT: return true;
        case Expression::Tag::CALL: {
            auto e = static_cast<CallExpr const *>(expr);
            if (e->is_external()) { return false; }
            return std::all_of(e->arguments().begin(), e->arguments().end(),
                               [](auto x) noexcept {
                                   return is_expr_statically_evaluated(x);
                               });
        }
        case Expression::Tag::CAST: {
            return is_expr_statically_evaluated(
                static_cast<CastExpr const *>(expr)->expression());
        }
        default: break;
    }
    return false;
}

// internalize an expression that is captured by a callable
const Expression *FunctionBuilder::_internalize(const Expression *expr) noexcept {
    if (expr == nullptr || expr->builder() == this) {
        return expr;
    }
    if (expr->type() == nullptr) {
        LUISA_ASSERT(expr->builder() == this,
                     "Cannot internalize expression with no type.");
        return expr;
    }
    auto is_static = is_expr_statically_evaluated(expr);
    // Note: we support "variable leaking" into caller to help graph-style
    //  shading system implementation but this can be dangerous. So we also
    //  print a warning message with stacktrace for users to check twice.
    if (_tag != Function::Tag::CALLABLE && !is_static) [[unlikely]] {
        // must be a leak from a callable, so postpone
        // the fix until the kernel is encoded
        LUISA_ASSERT(expr->tag() == Expression::Tag::REF,
                     "Leaked expression should be a reference.");
#ifdef NDEBUG// release build, relax the check
        LUISA_VERBOSE_WITH_LOCATION(
            "Leaking expression from callable to kernel. "
            "This might cause unexpected behavior. Please check.");
#else
        auto bt = luisa::backtrace();
        auto message = luisa::format(
            "Leaking expression from callable to kernel. "
            "This might cause unexpected behavior. Please check. [{}:{}]",
            __FILE__, __LINE__);
        for (auto &&frame : bt) {
            message.append("\n    ").append(luisa::to_string(frame));
        }
        LUISA_WARNING("{}", message);
#endif
        return expr;
    }
    // check if already internalized
    if (auto iter = _captured_external_variables.find(expr);
        iter != _captured_external_variables.end()) {
        return iter->second;
    }
    // check before internalizing
    if (!is_static) { check_expr_is_internalizable(expr); }
    // internalize the expression
    auto internalized = [this, is_static, external = expr]() noexcept -> const Expression * {
        auto mark_internalizer_argument = [this, external](auto expr) noexcept -> const Expression * {
            _internalizer_arguments.emplace(expr->variable(), external);
            return expr;
        };
        auto internalize_rvalue = [this, external, mark_internalizer_argument] {
            auto src = std::find(stack().rbegin(), stack().rend(), external->builder());
            LUISA_ASSERT(src != stack().rend(),
                         "Cannot internalize r-value expression "
                         "that is not on the stack.");
            return mark_internalizer_argument(argument(external->type()));
        };
        auto internalize_lvalue = [this, external, mark_internalizer_argument] {
            auto src = std::find(stack().rbegin(), stack().rend(), external->builder());
            auto on_stack = src != stack().rend();
            // if the external expression is not on the stack, we defer
            // the internalization until the full kernel is encoded
            if (!on_stack) {
                LUISA_ASSERT(external->tag() == Expression::Tag::REF,
                             "Cannot internalize non-reference "
                             "expression that is not on the stack.");
                return external;
            }
            return mark_internalizer_argument(reference(external->type()));
        };
        switch (external->tag()) {
            case Expression::Tag::MEMBER: {
                if (is_static || expr_is_lvalue_local_variable(external)) {
                    auto expr = static_cast<const MemberExpr *>(external);
                    auto self = _internalize(expr->self());
                    if (expr->is_swizzle()) {
                        LUISA_ASSERT(expr->swizzle_size() == 1u,
                                     "Cannot internalize r-value swizzle.");
                        return swizzle(expr->type(), self, 1u, expr->swizzle_code());
                    }
                    return member(expr->type(), self, expr->member_index());
                }
                return internalize_rvalue();
            }
            case Expression::Tag::ACCESS: {
                if (is_static || expr_is_lvalue_local_variable(external)) {
                    auto expr = static_cast<const AccessExpr *>(external);
                    auto range = _internalize(expr->range());
                    auto index = _internalize(expr->index());
                    return access(expr->type(), range, index);
                }
                return internalize_rvalue();
            }
            case Expression::Tag::LITERAL: {
                auto expr = static_cast<const LiteralExpr *>(external);
                return literal(expr->type(), expr->value());
            }
            case Expression::Tag::REF: {
                auto expr = static_cast<const RefExpr *>(external);
                auto v = expr->variable();
                switch (v.tag()) {
                    case Variable::Tag::LOCAL: [[fallthrough]];
                    case Variable::Tag::REFERENCE: return internalize_lvalue();
                    case Variable::Tag::BUFFER: return mark_internalizer_argument(buffer(v.type()));
                    case Variable::Tag::TEXTURE: return mark_internalizer_argument(texture(v.type()));
                    case Variable::Tag::BINDLESS_ARRAY: return mark_internalizer_argument(bindless_array());
                    case Variable::Tag::ACCEL: return mark_internalizer_argument(accel());
                    case Variable::Tag::THREAD_ID: [[fallthrough]];
                    case Variable::Tag::BLOCK_ID: [[fallthrough]];
                    case Variable::Tag::DISPATCH_ID: [[fallthrough]];
                    case Variable::Tag::DISPATCH_SIZE: [[fallthrough]];
                    case Variable::Tag::KERNEL_ID: [[fallthrough]];
                    case Variable::Tag::WARP_LANE_COUNT: [[fallthrough]];
                    case Variable::Tag::WARP_LANE_ID: [[fallthrough]];
                    case Variable::Tag::RASTER_OBJECT_ID: [[fallthrough]];
                    case Variable::Tag::RASTER_BARYCENTRICS: return _builtin(v.type(), v.tag());
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot internalize reference to {} variable {}.",
                    luisa::to_string(v.tag()), v.uid());
            }
            case Expression::Tag::CONSTANT: {
                auto expr = static_cast<const ConstantExpr *>(external);
                return constant(expr->data());
            }
            case Expression::Tag::UNARY: {
                if (is_static) {
                    auto e = static_cast<const UnaryExpr *>(external);
                    auto x = _internalize(e->operand());
                    return unary(e->type(), e->op(), x);
                }
                return internalize_rvalue();
            }
            case Expression::Tag::BINARY: {
                if (is_static) {
                    auto e = static_cast<const BinaryExpr *>(external);
                    auto lhs = _internalize(e->lhs());
                    auto rhs = _internalize(e->rhs());
                    return binary(e->type(), e->op(), lhs, rhs);
                }
                return internalize_rvalue();
            }
            case Expression::Tag::CALL: {
                if (is_static) {
                    auto e = static_cast<const CallExpr *>(external);
                    CallExpr::ArgumentList args;
                    args.reserve(e->arguments().size());
                    for (auto arg : e->arguments()) {
                        args.emplace_back(_internalize(arg));
                    }
                    if (e->is_builtin()) { return call(e->type(), e->op(), args, e->curve_basis_set()); }
                    if (e->is_custom()) { return call(e->type(), e->custom(), args); }
                    LUISA_ERROR_WITH_LOCATION("Invalid call expression to internalize.");
                }
                return internalize_rvalue();
            }
            case Expression::Tag::CAST: {// must be r-value
                if (is_static) {
                    auto e = static_cast<const CastExpr *>(external);
                    auto x = _internalize(e->expression());
                    return cast(e->type(), e->op(), x);
                }
                return internalize_rvalue();
            }
            case Expression::Tag::TYPE_ID: {
                auto expr = static_cast<const TypeIDExpr *>(external);
                return type_id(expr->data_type());
            }
            case Expression::Tag::STRING_ID: {
                auto expr = static_cast<const StringIDExpr *>(external);
                return string_id(luisa::string{expr->data()});
            }
            case Expression::Tag::CPUCUSTOM:
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot internalize CPU custom expression.");
            case Expression::Tag::GPUCUSTOM:
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot internalize GPU custom expression.");
            case Expression::Tag::FUNC_REF: LUISA_ERROR_WITH_LOCATION(
                "Cannot internalize function reference expression.");
        }
        LUISA_ERROR_WITH_LOCATION("Invalid expression to internalize.");
    }();
    // cache the internalized expression
    _captured_external_variables.emplace(expr, internalized);
    return internalized;
}

luisa::optional<uint8_t> FunctionBuilder::allowed_warp_size() const noexcept {
    if (_allowed_warp_size != 255) {
        return _allowed_warp_size;
    }
    return luisa::nullopt;
}

void FunctionBuilder::set_allowed_warp_size(uint8_t value) noexcept {
    switch (value) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
            _allowed_warp_size = value;
            break;
        default:
            LUISA_ERROR("Illegal warp size, must be 1, 2, 4, 8, 16, 32, 64, or 128");
            break;
    }
}
void FunctionBuilder::clear_allowed_warp_size() noexcept {
    _allowed_warp_size = 255;
}
bool FuncBuilderEqual::operator()(const FunctionBuilder *a,
                                  const FunctionBuilder *b) const noexcept {
    if (a != nullptr && b != nullptr) {
        return *a == *b;
    }
    return a == b;
}
size_t FuncBuilderHash::operator()(const FunctionBuilder *ptr, uint64_t hash) const noexcept {
    if (ptr) {
        if (hash == hash64_default_seed) [[likely]] {
            return ptr->hash();
        } else {
            return hash_value(ptr->hash(), hash);
        }
    } else {
        return 0;
    }
}

luisa::unordered_map<luisa::string, Function::FunctionAttribute> &FunctionBuilder::func_attributes() noexcept {
    return _func_attributes;
}

luisa::unordered_map<luisa::string, Function::FunctionAttribute> const &FunctionBuilder::func_attributes() const noexcept {
    return _func_attributes;
}
}// namespace luisa::compute::detail
