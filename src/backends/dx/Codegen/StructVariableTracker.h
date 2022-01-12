#pragma once
#include <vstl/Common.h>
#include <EASTL/vector.h>
namespace toolhub::directx {
class StructVariableTracker {
    eastl::vector<
        vstd::HashMap<
            vstd::string,
            vstd::HashMap<
                vstd::string,
                std::pair<vstd::string, bool>>>>
        stacks;
    void SetStack(size_t stackCount);

public:
    StructVariableTracker();
    ~StructVariableTracker();
    void Clear();
    void RemoveStack(vstd::string &str);
    vstd::string_view CreateTempVar(
        size_t stackCount,
        vstd::string &str,
        vstd::string_view structName,
        vstd::string_view memberName,
        vstd::string_view tempVarName,
        bool transformed);
    void ClearTempVar(
        size_t stackCount,
        vstd::string &str,
        vstd::string_view structName);

    vstd::string_view GetStructMemberName(
        size_t stackCount,
        vstd::string_view structName,
        vstd::string_view memberName);
};
}// namespace toolhub::directx