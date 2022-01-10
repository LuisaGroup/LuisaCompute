#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include <ast/expression.h>
#include <ast/statement.h>
using namespace luisa::compute;
namespace toolhub::directx {
struct StructVariable {
    static constexpr vbyte OFFSET_NPOS = std::numeric_limits<vbyte>::max();
    vstd::string name;
    vbyte boolOffset;
    vbyte boolVecDim;
    StructVariable(
        vstd::string &&name,
        vbyte boolVecDim = 0,
        vbyte boolOffset = OFFSET_NPOS) : name(std::move(name)), boolVecDim(boolVecDim), boolOffset(boolOffset){};
};
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
    vstd::vector<StructVariable> structVars;
    vstd::vector<vstd::variant<StructureType, StructGenerator *>> structTypes;
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
    static void ProvideAlignVariable(size_t tarAlign, size_t &structSize, size_t &alignCount, vstd::string& structDesc);
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
    void GetMemberExpr(
        uint memberIndex,
        Type const *type,
        vstd::function<void()> const &printMember,
        vstd::string &str);
    void SetMemberExpr(
        uint memberIndex,
        Type const *type,
        vstd::function<void()> const &printMember,
        vstd::function<void()> const &printExpr,
        vstd::string &str);
};
}// namespace toolhub::directx