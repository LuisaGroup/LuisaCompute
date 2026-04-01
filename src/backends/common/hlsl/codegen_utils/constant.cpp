// Constant Handling

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"

namespace lc::hlsl {

// Get/generate constant name
bool CodegenUtility::GetConstName(uint64 hash, ConstantData const &data, vstd::StringBuilder &str) const {
    auto constCount = opt->GetConstCount(hash);
    str << "c";
    vstd::to_string((constCount.first), str);
    return constCount.second;
}

}// namespace lc::hlsl
