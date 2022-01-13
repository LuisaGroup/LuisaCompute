#pragma vengine_package vengine_directx
#include <Codegen/StructVariableTracker.h>
namespace toolhub::directx {
StructVariableTracker::StructVariableTracker() {}
StructVariableTracker::~StructVariableTracker() {}
void StructVariableTracker::RemoveStack(vstd::string &str) {
    if (stacks.empty()) return;
    auto last = stacks.end() - 1;
    for (auto &&stct : *last) {
        for (auto &&kv : stct.second) {
            if (kv.second.second) {
                str << stct.first
                    << '.'
                    << kv.first
                    << "=SetBool("sv
                    << kv.second.first
                    << ");\n"sv;
                kv.second.second = false;
            }
        }
    }
    stacks.erase(last);
}
void StructVariableTracker::Clear() {
    stacks.clear();
}
void StructVariableTracker::SetStack(size_t stackCount) {
    stackCount++;
    while (stacks.size() < stackCount) {
        stacks.emplace_back();
    }
}

vstd::string_view StructVariableTracker::CreateTempVar(
    size_t stackCount,
    vstd::string &str,
    vstd::string_view structName,
    vstd::string_view memberName,
    vstd::string_view tempVarName,
    bool transformed) {
    vstd::string_view tmpName;
    SetStack(stackCount);
    auto &&map = stacks[stackCount];
    auto ite = map.Emplace(structName);
    auto subIte = ite.Value().TryEmplace(memberName);
    if (subIte.second) {
        str << "bool4 "sv << tempVarName << "=GetBool("sv << structName << '.' << memberName << ");\n"sv;
        subIte.first.Value().first = tempVarName;
        tmpName = tempVarName;
    } else {
        tmpName = subIte.first.Value().first;
    }
    if (transformed) subIte.first.Value().second = transformed;
    return tmpName;
}
void StructVariableTracker::ClearTempVar(
    size_t stackCount,
    vstd::string &str,
    vstd::string_view structName) {
    if (stacks.empty()) return;
    stackCount = std::min(stackCount, stacks.size() - 1);
    do {
        auto &&map = stacks[stackCount];
        auto ite = map.Find(structName);
        if (!ite) {
            continue;
        }
        for (auto &&kv : ite.Value()) {
            if (kv.second.second) {
                str << ite.Key()
                    << '.'
                    << kv.first
                    << "=SetBool("sv
                    << kv.second.first
                    << ");\n"sv;
                kv.second.second = false;
            }
        }
        return;
    } while (stackCount != 0);
}
vstd::string_view StructVariableTracker::GetStructMemberName(
    size_t stackCount,
    vstd::string_view structName,
    vstd::string_view memberName) {
    if (stacks.size() <= stackCount) return memberName;
    auto ite = stacks[stackCount].Find(structName);
    if (!ite) return memberName;
    auto subIte = ite.Value().Find(memberName);
    if (!subIte) return memberName;
    return subIte.Value().first;
}
}// namespace toolhub::directx