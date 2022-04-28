#pragma once
#include <vstl/Common.h>
#include <Codegen/DxCodegen.h>
#include <Codegen/StructGenerator.h>
#include <compile/definition_analysis.h>
namespace toolhub::directx {

struct CodegenStackData : public vstd::IOperatorNewBase {
    int64 scopeCount = -1;
    vstd::HashMap<Type const *, uint64> structTypes;
    vstd::HashMap<uint64, uint64> constTypes;
    vstd::HashMap<uint64, uint64> funcTypes;
    vstd::HashMap<Type const *, vstd::unique_ptr<StructGenerator>> customStruct;
    vstd::HashMap<Type const *, uint64> bindlessBufferTypes;
    vstd::HashMap<uint> arguments;
    bool isKernel = false;
    vstd::vector<StructGenerator *> customStructVector;
    uint64 count = 0;
    uint64 constCount = 0;
    uint64 funcCount = 0;
    uint64 tempCount = 0;
    uint64 bindlessBufferCount = 0;
    uint64 structCount = 0;
    DefinitionAnalysis analyzer;
    
    vstd::function<StructGenerator *(Type const *)> generateStruct;
    StructGenerator *rayDesc = nullptr;
    StructGenerator *hitDesc = nullptr;
    vstd::HashMap<vstd::string, vstd::string> structReplaceName;
    vstd::HashMap<uint64_t> generatedConstants;
    DefinitionAnalysis::VariableSet sharedVariable;
    DefinitionAnalysis::VariableSet allVariables;
    CodegenStackData();
    void Clear();
    uint AddBindlessType(Type const *type);
    StructGenerator *CreateStruct(Type const *t);
    uint64 GetConstCount(uint64 data);
    uint64 GetFuncCount(uint64 data);
    uint64 GetTypeCount(Type const *t);
    ~CodegenStackData();
    static vstd::unique_ptr<CodegenStackData> Allocate();
    static void DeAllocate(vstd::unique_ptr<CodegenStackData>&& v);
};
}// namespace toolhub::directx