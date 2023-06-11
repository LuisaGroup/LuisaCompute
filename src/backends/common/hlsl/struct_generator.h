#pragma once
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/ast/function.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include "string_builder.h"
namespace lc::hlsl {
class CodegenUtility;
using namespace luisa::compute;
class StructGenerator : public vstd::IOperatorNewBase {
public:
    using Callback = vstd::function<void(Type const *)>;

private:
    Type const *structureType{nullptr};
    CodegenUtility *util;
    // vstd::vector<vstd::variant<StructureType, StructGenerator *>> structTypes;
    vstd::StringBuilder structDesc;
    vstd::string structName;
    size_t idx;
    void InitAsStruct(
        vstd::span<Type const *const> const &vars,
        size_t structIdx,
        Callback const &visitor);
    void InitAsArray(
        Type const *structureType,
        size_t structIdx,
        Callback const &visitor);

public:
    static void ProvideAlignVariable(size_t tarAlign, size_t &align, size_t &structSize, vstd::StringBuilder &structDesc);
    vstd::string_view GetStructDesc() const { return structDesc.view(); }
    vstd::string_view GetStructName() const { return structName; }
    void SetStructName(vstd::string &&name) {
        structName = std::move(name);
    }
    size_t Index() const { return idx; }
    Type const *GetType() const noexcept { return structureType; }
    StructGenerator(
        Type const *structureType,
        size_t structIdx,
        CodegenUtility *util);
    void Init(Callback const &visitor);
    ~StructGenerator();
};
}// namespace lc::hlsl
