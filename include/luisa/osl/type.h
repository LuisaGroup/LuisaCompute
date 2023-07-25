//
// Created by Mike Smith on 2023/7/24.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::osl {

class Type {

public:
    enum struct Tag {
        SIMPLE,
        STRUCT,
        CLOSURE,
    };

private:
    Tag _tag;

protected:
    explicit Type(Tag tag) noexcept : _tag{tag} {}

public:
    virtual ~Type() noexcept = default;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual luisa::string_view identifier() const noexcept = 0;

    // for debugging
    [[nodiscard]] virtual luisa::string dump() const noexcept {
        return luisa::string{identifier()};
    }
};

class LC_OSL_API SimpleType final : public Type {

public:
    enum struct Primitive {
        VOID,
        INT,
        FLOAT,
        POINT,
        NORMAL,
        VECTOR,
        COLOR,
        MATRIX,
        STRING,
    };

private:
    Primitive _primitive;

public:
    explicit SimpleType(Primitive primitive) noexcept
        : Type{Tag::SIMPLE}, _primitive{primitive} {}
    // disable copy and move
    SimpleType(const SimpleType &) noexcept = delete;
    SimpleType(SimpleType &&) noexcept = delete;
    SimpleType &operator=(const SimpleType &) noexcept = delete;
    SimpleType &operator=(SimpleType &&) noexcept = delete;
    ~SimpleType() noexcept override = default;
    [[nodiscard]] auto primitive() const noexcept { return _primitive; }
    [[nodiscard]] luisa::string_view identifier() const noexcept override;
};

class LC_OSL_API StructType final : public Type {

public:
    struct Field {
        luisa::string name;
        const Type *type;
    };

private:
    luisa::string _identifier;
    luisa::vector<Field> _fields;

public:
    StructType(luisa::string identifier, luisa::vector<Field> fields = {}) noexcept;
    ~StructType() noexcept override = default;
    void set_fields(luisa::vector<Field> fields) noexcept { _fields = std::move(fields); }
    [[nodiscard]] luisa::string_view identifier() const noexcept override { return _identifier; }
    [[nodiscard]] auto fields() noexcept { return luisa::span{_fields}; }
    [[nodiscard]] auto fields() const noexcept { return luisa::span{_fields}; }
};

class LC_OSL_API ClosureType final : public Type {

private:
    const Type *_gentype;

public:
    explicit ClosureType(const Type *gentype) noexcept;
    ~ClosureType() noexcept override = default;
    [[nodiscard]] luisa::string_view identifier() const noexcept override;
    [[nodiscard]] auto gentype() const noexcept { return _gentype; }
};

}// namespace luisa::compute::osl
