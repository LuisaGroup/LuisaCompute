#pragma once
#include <vstl/common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <HLSL/string_builder.h>
using namespace luisa::compute;
namespace toolhub::directx {
class StructGenerator : public vstd::IOperatorNewBase {
public:
    using Callback = vstd::function<void(Type const *)>;

private:
    Type const *structureType{nullptr};
    // vstd::vector<vstd::variant<StructureType, StructGenerator *>> structTypes;
    vstd::StringBuilder structDesc;
    vstd::string structName;
    size_t idx;
    void InitAsStruct(
        vstd::Iterator<Type const *const> const &vars,
        size_t structIdx,
        Callback const &visitor);
    void InitAsArray(
        Type const *structureType,
        size_t structIdx,
        Callback const &visitor);

public:
    static void ProvideAlignVariable(size_t tarAlign, size_t& align, size_t &structSize, vstd::StringBuilder &structDesc);
    vstd::string_view GetStructDesc() const { return structDesc.view(); }
    vstd::string_view GetStructName() const { return structName; }
    void SetStructName(vstd::string &&name) {
        structName = std::move(name);
    }
    size_t Index() const { return idx; }
    Type const *GetType() const noexcept { return structureType; }
    StructGenerator(
        Type const *structureType,
        size_t structIdx);
    void Init(Callback const &visitor);
    ~StructGenerator();
};
}// namespace toolhub::directx