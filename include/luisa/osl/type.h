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
        ARRAY,
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

class LC_OSL_API ArrayType final : public Type {

private:
    const Type *_element;
    size_t _length;
    luisa::string _identifier;

public:
    ArrayType(const Type *element, size_t length) noexcept;
    ~ArrayType() noexcept override = default;
    [[nodiscard]] luisa::string_view identifier() const noexcept override { return _identifier; }
    [[nodiscard]] auto element() const noexcept { return _element; }
    [[nodiscard]] auto length() const noexcept { return _length; }
    [[nodiscard]] auto is_unbounded() const noexcept { return _length == 0u; }
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
