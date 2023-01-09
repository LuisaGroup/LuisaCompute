#pragma once
#include <vstl/common.h>
#include "dx_codegen.h"
#include "struct_generator.h"
namespace toolhub::directx {

struct CodegenStackData : public vstd::IOperatorNewBase {
    luisa::compute::Function kernel;
    vstd::unordered_map<Type const *, uint64> structTypes;
    vstd::unordered_map<uint64, uint64> constTypes;
    vstd::unordered_map<void const*, uint64> funcTypes;
    vstd::unordered_map<Type const *, vstd::unique_ptr<StructGenerator>> customStruct;
    vstd::unordered_map<Type const *, uint64> bindlessBufferTypes;
    vstd::unordered_map<uint, uint> arguments;
    enum class FuncType : uint8_t{
        Kernel,
        Vert,
        Pixel,
        Callable
    };
    FuncType funcType;
    bool isRaster = false;
    bool isPixelShader = false;
    bool pixelFirstArgIsStruct = false;
    vstd::vector<StructGenerator *> customStructVector;
    uint64 count = 0;
    uint64 constCount = 0;
    uint64 funcCount = 0;
    uint64 tempCount = 0;
    uint64 bindlessBufferCount = 0;
    uint64 structCount = 0;
    int64_t appdataId = -1;
    int64 scopeCount = -1;
    
    vstd::function<StructGenerator *(Type const *)> generateStruct;
    StructGenerator *rayDesc = nullptr;
    StructGenerator *hitDesc = nullptr;
    vstd::unordered_map<vstd::string, vstd::string> structReplaceName;
    vstd::unordered_map<uint64, Variable> sharedVariable;
    Expression const *tempSwitchExpr;
    size_t tempSwitchCounter = 0;
    CodegenStackData();
    void Clear();
    uint AddBindlessType(Type const *type);
    StructGenerator *CreateStruct(Type const *t);
    std::pair<uint64, bool> GetConstCount(uint64 data);
    uint64 GetFuncCount(void const* data);
    uint64 GetTypeCount(Type const *t);
    ~CodegenStackData();
    static vstd::unique_ptr<CodegenStackData> Allocate();
    static void DeAllocate(vstd::unique_ptr<CodegenStackData>&& v);
    // static bool& ThreadLocalSpirv();
};
}// namespace toolhub::directx