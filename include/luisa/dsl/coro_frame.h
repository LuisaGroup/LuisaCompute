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
    static constexpr size_t reserved_field_count = 3u;

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

    [[nodiscard]] auto frame_field_type(size_t index) const noexcept {
        switch (index) {
            case 0u: return Type::of<uint3>();
            case 1u: return Type::of<uint>();
            case 2u: return Type::of<uint>();
            default:
                LUISA_ASSERT(index < frame_field_count(), "CoroFrame field index out of range.");
                return _fields[index - reserved_field_count].type;
        }
    }

    [[nodiscard]] auto frame_alignment() const noexcept -> size_t {
        auto alignment = Type::of<uint3>()->alignment();
        alignment = std::max(alignment, Type::of<uint>()->alignment());
        for (auto i = 0u; i < field_count(); i++) {
            alignment = std::max(alignment, field(i).type->alignment());
        }
        return alignment;
    }

    [[nodiscard]] auto frame_type() const noexcept -> const Type * {
        luisa::vector<const Type *> members;
        members.reserve(frame_field_count());
        members.emplace_back(Type::of<uint3>());
        members.emplace_back(Type::of<uint>());
        members.emplace_back(Type::of<uint>());
        for (auto i = 0u; i < field_count(); i++) {
            members.emplace_back(field(i).type);
        }
        return Type::structure(frame_alignment(), members);
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
    UInt3 coro_id;
    UInt target_token;
    UInt skip_flag;

    CoroFrame(const CoroFrameDesc *desc, const Expression *expression) noexcept
        : _desc{desc},
          _type{desc->frame_type()},
          _expression{expression},
          coro_id{detail::FunctionBuilder::current()->member(Type::of<uint3>(), _expression, 0u)},
          target_token{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 1u)},
          skip_flag{detail::FunctionBuilder::current()->member(Type::of<uint>(), _expression, 2u)} {}

    explicit CoroFrame(const CoroFrameDesc *desc) noexcept
        : CoroFrame{desc, detail::FunctionBuilder::current()->local(desc->frame_type())} {}

    [[nodiscard]] static auto create(const CoroFrameDesc *desc) noexcept {
        return CoroFrame{desc};
    }

    [[nodiscard]] auto desc() const noexcept { return _desc; }
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    template<typename T>
    [[nodiscard]] auto get(size_t index) const noexcept {
        LUISA_ASSERT(index < _desc->field_count(), "CoroFrame field index out of range.");
        auto field_index = index + 3u;
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
