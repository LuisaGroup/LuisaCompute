#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
using namespace luisa::compute;
namespace toolhub::directx {
class StructureType : public vstd::IOperatorNewBase {

public:
    enum class Tag : vbyte {
        Scalar,
        Vector,
        Matrix
    };

private:
    vbyte mDimension;
    Tag mTag;
    StructureType(Tag t, vbyte d)
        : mTag(t), mDimension(d) {}

public:
    static StructureType GetScalar();
    static StructureType GetVector(vbyte dim);
    static StructureType GetMatrix(vbyte dim);
    vbyte dimension() const { return mDimension; }
    Tag tag() const { return mTag; }
    size_t size() const;
    size_t align() const;
};
class StructGenerator : public vstd::IOperatorNewBase {
    Type const *strutureType;
    vstd::vector<std::pair<vstd::string, vstd::variant<StructureType, StructGenerator *>>> structTypes;
    vstd::string structDesc;
    vstd::string structName;
    size_t structSize = 0;
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
    vstd::string const& GetStructVar(uint idx) const{
        return structTypes[idx].first;
    }
    static void ProvideAlignVariable(size_t tarAlign, size_t &structSize, size_t &alignCount, vstd::string &structDesc);
    vstd::string_view GetStructDesc() const { return structDesc; }
    vstd::string_view GetStructName() const { return structName; }
    size_t GetStructSize() const { return structSize; }
    size_t Index() const { return idx; }
    StructGenerator(
        Type const *structureType,
        size_t structIdx,
        vstd::function<StructGenerator *(Type const *)> const &visitor);
    StructGenerator(
        vstd::Iterator<Type const *const> const &vars,
        size_t structIdx,
        vstd::function<StructGenerator *(Type const *)> const &visitor);

    ~StructGenerator();
};
}// namespace toolhub::directx