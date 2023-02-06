#pragma once
#include <vstl/common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <vstl/string_builder.h>
using namespace luisa::compute;
namespace toolhub::directx {
class StructureType : public vstd::IOperatorNewBase {

public:
    enum class Tag : uint8_t {
        Scalar,
        Vector,
        Matrix
    };

private:
    uint8_t mDimension;
    Tag mTag;
    StructureType(Tag t, uint8_t d)
        : mDimension(d), mTag(t) {}

public:
    static StructureType GetScalar();
    static StructureType GetVector(uint8_t dim);
    static StructureType GetMatrix(uint8_t dim);
    uint8_t dimension() const { return mDimension; }
    Tag tag() const { return mTag; }
    //size_t size() const;
    //  size_t align() const;
};
class StructGenerator : public vstd::IOperatorNewBase {
    Type const *structureType{nullptr};
    vstd::vector<std::pair<vstd::string, vstd::variant<StructureType, StructGenerator *>>> structTypes;
    vstd::StringBuilder structDesc;
    vstd::string structName;
    size_t alignCount = 0;
    size_t idx;
    void InitAsStruct(
        vstd::Iterator<Type const *const> const &vars,
        size_t structIdx,
        vstd::function<StructGenerator *(Type const *)> const &visitor);
    void InitAsArray(
        Type const *structureType,
        size_t structIdx,
        vstd::function<StructGenerator *(Type const *)> const &visitor);

public:
    vstd::string const &GetStructVar(uint idx) const {
        return structTypes[idx].first;
    }
    static void ProvideAlignVariable(size_t tarAlign, size_t &structSize, size_t &alignCount, vstd::StringBuilder &structDesc);
    vstd::string_view GetStructDesc() const { return structDesc.view(); }
    vstd::string_view GetStructName() const { return structName; }
    void SetStructName(vstd::string &&name) {
        structName = std::move(name);
    }
    size_t Index() const { return idx; }
    size_t AlignCount() const { return alignCount; }
    Type const *GetType() const noexcept { return structureType; }
    StructGenerator(
        Type const *structureType,
        size_t structIdx);
    void Init(vstd::function<StructGenerator *(Type const *)> const &visitor);
    ~StructGenerator();
};
}// namespace toolhub::directx