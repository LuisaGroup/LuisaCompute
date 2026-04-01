// Resource & Utility Management

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"

namespace lc::hlsl {

// Generate unique temporary variable names
vstd::StringBuilder CodegenUtility::GetNewTempVarName() const {
    vstd::StringBuilder name;
    name << "tmp"sv;
    vstd::to_string(opt->tempCount, name);
    opt->tempCount++;
    return name;
}

// Constructor
CodegenUtility::CodegenUtility() {
    attributes.try_emplace("position", "POSITION", nullptr);
    attributes.try_emplace("normal", "NORMAL", nullptr);
    attributes.try_emplace("tangent", "TANGENT", nullptr);
    attributes.try_emplace("color", "COLOR", nullptr);
    attributes.try_emplace("uv0", "TEXCOORD0", nullptr);
    attributes.try_emplace("uv1", "TEXCOORD1", nullptr);
    attributes.try_emplace("uv2", "TEXCOORD2", nullptr);
    attributes.try_emplace("uv3", "TEXCOORD3", nullptr);
    attributes.try_emplace("vertex_id", "SV_VertexID", Type::of<uint>());
    attributes.try_emplace("instance_id", "SV_InstanceID", Type::of<uint>());
    attributes.try_emplace("is_front_face", "SV_IsFrontFace", Type::of<bool>());
}

// Destructor
CodegenUtility::~CodegenUtility() = default;

}// namespace lc::hlsl
