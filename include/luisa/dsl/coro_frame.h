#pragma once

#include <cstddef>
#include <string_view>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/operators.h>
#include <luisa/xir/passes/coro_materialize.h>

namespace luisa::compute {

struct CoroFrameIdProxy;

namespace detail {

template<>
struct expr_value_impl<CoroFrameIdProxy> {
    using type = uint3;
};

template<>
struct is_dsl_impl<CoroFrameIdProxy> : std::true_type {};

}// namespace detail

struct CoroFrameIdProxy {
public:
    UInt x;
    UInt y;
    UInt z;

    CoroFrameIdProxy(const Expression *x, const Expression *y, const Expression *z) noexcept
        : x{x}, y{y}, z{z} {}

    [[nodiscard]] auto expression() const noexcept {
        return detail::FunctionBuilder::current()->call(
            Type::of<uint3>(), CallOp::MAKE_UINT3,
            {x.expression(), y.expression(), z.expression()});
    }

    [[nodiscard]] operator Expr<uint3>() const noexcept {
        return Expr<uint3>{expression()};
    }

    template<typename Rhs>
        requires is_vector3_expr_v<Rhs> &&
                 std::same_as<vector_expr_element_t<Rhs>, uint>
    void operator=(Rhs &&rhs) & noexcept {
        auto value = Expr<expr_value_t<Rhs>>{detail::extract_expression(std::forward<Rhs>(rhs))};
        x = value.x;
        y = value.y;
        z = value.z;
    }
};

inline namespace dsl {

template<typename Rhs>
    requires is_vector3_expr_v<Rhs> &&
             std::same_as<vector_expr_element_t<Rhs>, uint>
void assign(CoroFrameIdProxy &lhs, Rhs &&rhs) noexcept {
    lhs = std::forward<Rhs>(rhs);
}

}// namespace dsl

struct CoroFrameDesc {
    struct FieldDescriptor {
        luisa::string name;
        const Type *type{nullptr};
        size_t offset{0u};
        size_t size{0u};
    };

private:
    luisa::vector<FieldDescriptor> _fields;
    size_t _total_size{0u};

public:
    static constexpr size_t reserved_field_count = 4u;

    void add_field(luisa::string name, const Type *type) noexcept {
        auto alignment = type->alignment();
        auto size = type->size();
        auto offset = (_total_size + alignment - 1u) / alignment * alignment;
        _fields.emplace_back(FieldDescriptor{
            std::move(name),
            type,
            offset,
            size});
        _total_size = offset + size;
        _total_size = (_total_size + alignment - 1u) / alignment * alignment;
    }

    [[nodiscard]] auto field_count() const noexcept { return _fields.size(); }

    [[nodiscard]] const FieldDescriptor &field(size_t index) const noexcept {
        return _fields[index];
    }

    [[nodiscard]] const FieldDescriptor *field(luisa::string_view name) const noexcept {
        for (auto &field : _fields) {
            if (field.name == name) {
                return &field;
            }
        }
        return nullptr;
    }

    [[nodiscard]] auto total_size() const noexcept { return _total_size; }

    [[nodiscard]] auto frame_field_count() const noexcept { return reserved_field_count + _fields.size(); }

    [[nodiscard]] auto frame_field_name(size_t index) const noexcept {
        switch (index) {
            case 0u: return luisa::string_view{"coro_id.x"};
            case 1u: return luisa::string_view{"coro_id.y"};
            case 2u: return luisa::string_view{"coro_id.z"};
            case 3u: return luisa::string_view{"target_token"};
            default:
                LUISA_ASSERT(index < frame_field_count(), "CoroFrame field index out of range.");
                return luisa::string_view{_fields[index - reserved_field_count].name};
        }
    }

    [[nodiscard]] auto frame_field_type(size_t index) const noexcept {
        switch (index) {
            case 0u:
            case 1u:
            case 2u:
            case 3u: return Type::of<uint>();
            default:
                LUISA_ASSERT(index < frame_field_count(), "CoroFrame field index out of range.");
                return _fields[index - reserved_field_count].type;
        }
    }

    [[nodiscard]] auto frame_alignment() const noexcept -> size_t {
        auto alignment = Type::of<uint>()->alignment();
        for (auto i = 0u; i < field_count(); i++) {
            alignment = std::max(alignment, field(i).type->alignment());
        }
        return alignment;
    }

    [[nodiscard]] auto frame_type() const noexcept -> const Type * {
        luisa::vector<const Type *> members;
        members.reserve(frame_field_count());
        for (auto i = 0u; i < reserved_field_count; i++) {
            members.emplace_back(Type::of<uint>());
        }
        for (auto i = 0u; i < field_count(); i++) {
            members.emplace_back(field(i).type);
        }
        return Type::structure(frame_alignment(), members);
    }

    [[nodiscard]] auto frame_field_offset(size_t index) const noexcept {
        LUISA_ASSERT(index < frame_field_count(), "CoroFrame field index out of range.");
        auto offset = size_t{0u};
        for (auto i = 0u; i <= index; i++) {
            auto *type = frame_field_type(i);
            auto alignment = type->alignment();
            offset = (offset + alignment - 1u) / alignment * alignment;
            if (i == index) { return offset; }
            offset += type->size();
        }
        return offset;
    }

    [[nodiscard]] auto dump() const noexcept {
        luisa::string s;
        s.append(luisa::format("Frame: fields={} payload={}B struct={}B alignment={}B\n",
                               frame_field_count(), total_size(), frame_type()->size(), frame_alignment()));
        for (auto i = 0u; i < frame_field_count(); i++) {
            auto *type = frame_field_type(i);
            s.append(luisa::format("  Field {}: {} type={} offset={} size={} align={}\n",
                                   i, frame_field_name(i), type->description(),
                                   frame_field_offset(i), type->size(), type->alignment()));
        }
        return s;
    }

    void from_materialize_info(const xir::CoroMaterializeInfo &info) noexcept {
        // Clear existing fields
        _fields.clear();
        _total_size = 0u;

        // Collect user fields sorted by their frame index
        luisa::vector<std::pair<luisa::string, size_t>> sorted_fields;
        for (const auto &[name, field_idx] : info.name_to_field) {
            sorted_fields.emplace_back(name, field_idx);
        }
        luisa::sort(sorted_fields.begin(), sorted_fields.end(),
                    [](const auto &a, const auto &b) noexcept {
                        return a.second < b.second;
                    });

        // Add user fields in order
        for (const auto &[name, field_idx] : sorted_fields) {
            auto type_it = info.name_to_type.find(name);
            if (type_it != info.name_to_type.end()) {
                add_field(name, type_it->second);
            }
        }
    }
};

struct CoroFrame {
private:
    const CoroFrameDesc *_desc{nullptr};
    const Type *_type{nullptr};
    const Expression *_expression{nullptr};

    [[nodiscard]] auto _field_index(luisa::string_view name) const noexcept -> size_t {
        for (auto i = 0u; i < _desc->field_count(); i++) {
            if (_desc->field(i).name == name) {
                return i;
            }
        }
        return static_cast<size_t>(-1);
    }

public:
    static constexpr uint TERMINAL_TOKEN = 0xFFFFFFFFu;
    UInt coro_id_x;
    UInt coro_id_y;
    UInt coro_id_z;
    UInt target_token;
    CoroFrameIdProxy coro_id;

    CoroFrame(const CoroFrameDesc *desc, const Expression *expression) noexcept
        : _desc{desc},
          _type{desc->frame_type()},
          _expression{expression},
          coro_id_x{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 0u)},
          coro_id_y{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 1u)},
          coro_id_z{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 2u)},
          target_token{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 3u)},
          coro_id{coro_id_x.expression(), coro_id_y.expression(), coro_id_z.expression()} {}

    explicit CoroFrame(const CoroFrameDesc *desc) noexcept
        : CoroFrame{desc, detail::FunctionBuilder::current()->local(desc->frame_type())} {}

    [[nodiscard]] static auto create(const CoroFrameDesc *desc) noexcept {
        auto *fb = detail::FunctionBuilder::current();
        auto *expr = fb->local(desc->frame_type());
        fb->assign(expr, fb->call(desc->frame_type(), CallOp::ZERO, {}));
        return CoroFrame{desc, expr};
    }

    [[nodiscard]] auto desc() const noexcept { return _desc; }
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    template<typename T>
    [[nodiscard]] auto get(size_t index) const noexcept {
        LUISA_ASSERT(index < _desc->field_count(), "CoroFrame field index out of range.");
        auto field_index = index + CoroFrameDesc::reserved_field_count;
        auto field = _desc->field(index);
        LUISA_ASSERT(field.type == Type::of<T>(), "CoroFrame field type mismatch.");
        return Var<T>{detail::FunctionBuilder::current()->member(Type::of<T>(), _expression, field_index)};
    }

    template<typename T>
    [[nodiscard]] auto get(luisa::string_view name) const noexcept {
        auto index = _field_index(name);
        LUISA_ASSERT(index != static_cast<size_t>(-1), "CoroFrame field not found.");
        return get<T>(index);
    }

    [[nodiscard]] auto is_terminated() const noexcept {
        return target_token == def(TERMINAL_TOKEN);
    }
};

}// namespace luisa::compute
