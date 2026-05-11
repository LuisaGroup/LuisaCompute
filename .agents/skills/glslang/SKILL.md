# glslang SPIR-V Usage

Khronos reference compiler SPIR-V components located in `src/ext/glslang/SPIRV`.

## Headers

```cpp
#include "SPIRV/SpvBuilder.h"
#include "SPIRV/spvIR.h"
#include "SPIRV/GlslangToSpv.h"
#include "SPIRV/disassemble.h"
```

## SpvBuilder Lifecycle

`spv::Builder` owns one SPIR-V module. Thread-safe internal IR.

```cpp
spv::SpvBuildLogger logger;
spv::Builder builder(spv::Spv_1_5, 0, &logger);

builder.setSource(spv::SourceLanguage::GLSL, 450);
builder.setMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
builder.addCapability(spv::Capability::Shader);

// ... build module ...

std::vector<unsigned int> spirv;
builder.dump(spirv);
```

## Module Setup

```cpp
builder.setSource(spv::SourceLanguage::GLSL, 450);
builder.setDebugMainSourceFile("shader.frag");
builder.setDebugSourceLocation(10, "shader.frag");

builder.addCapability(spv::Capability::Shader);
builder.addExtension("SPV_KHR_ray_tracing");

builder.setMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);

// Imports (e.g., GLSL.std.450)
spv::Id glsl450 = builder.import("GLSL.std.450");
```

## Types

All type creation is canonicalized (returns existing type if already made).

```cpp
spv::Id voidTy   = builder.makeVoidType();
spv::Id boolTy   = builder.makeBoolType();
spv::Id int32Ty  = builder.makeIntType(32);
spv::Id uint32Ty = builder.makeUintType(32);
spv::Id uint64Ty = builder.makeUint64Constant(0); // type via constant helper
spv::Id floatTy  = builder.makeFloatType(32);
spv::Id doubleTy = builder.makeFloatType(64);
spv::Id halfTy   = builder.makeFloatType(16);
spv::Id bfloat16 = builder.makeBFloat16Type();
spv::Id float8e5 = builder.makeFloatE5M2Type();
spv::Id float8e4 = builder.makeFloatE4M3Type();

spv::Id vec4Ty   = builder.makeVectorType(floatTy, 4);
spv::Id mat4x4Ty = builder.makeMatrixType(floatTy, 4, 4);

spv::Id arrTy    = builder.makeArrayType(floatTy, builder.makeUintConstant(16), 0);
spv::Id runArrTy = builder.makeRuntimeArray(floatTy);

std::vector<spv::Id> members = { floatTy, int32Ty };
std::vector<spv::StructMemberDebugInfo> memberDebug;
spv::Id structTy = builder.makeStructType(members, memberDebug, "MyStruct", false);

spv::Id ptrTy    = builder.makePointer(spv::StorageClass::Function, floatTy);
spv::Id fwdPtrTy = builder.makeForwardPointer(spv::StorageClass::PhysicalStorageBufferEXT);
spv::Id untypedPtr = builder.makeUntypedPointer(spv::StorageClass::StorageBuffer);

spv::Id fnTy     = builder.makeFunctionType(voidTy, { floatTy, int32Ty });

spv::Id imgTy    = builder.makeImageType(floatTy, spv::Dim::Dim2D, false, false, false,
                                          1, spv::ImageFormat::Rgba32f, "texture2D");
spv::Id sampledImgTy = builder.makeSampledImageType(imgTy, "sampler2D");
spv::Id samplerTy    = builder.makeSamplerType("sampler");

// Acceleration structure / ray query / hit object
spv::Id asTy = builder.makeAccelerationStructureType();
spv::Id rqTy = builder.makeRayQueryType();
spv::Id hoTy = builder.makeHitObjectEXTType();

// Cooperative matrix / vector / tensor
spv::Id coopMatTy = builder.makeCooperativeMatrixTypeKHR(floatTy, scopeId, rowsId, colsId, useId);
spv::Id coopVecTy = builder.makeCooperativeVectorTypeNV(floatTy, componentsId);
spv::Id tensorTy  = builder.makeTensorTypeARM(floatTy, rankId);

// Generic custom type
std::vector<spv::IdImmediate> ops;
ops.emplace_back(true, someId);
spv::Id genericTy = builder.makeGenericType(spv::Op::OpType... , ops);
```

## Type Queries

```cpp
spv::Id typeId = builder.getTypeId(resultId);
spv::Op opCode = builder.getOpCode(id);
spv::Op cls    = builder.getTypeClass(typeId);
bool isPtr     = builder.isPointer(id);
bool isScalar  = builder.isScalar(id);
bool isVec     = builder.isVector(id);
bool isMat     = builder.isMatrix(id);
bool isArray   = builder.isArrayType(typeId);
bool isStruct  = builder.isStructType(typeId);
bool isImage   = builder.isImageType(typeId);
bool isSampler = builder.isSamplerType(typeId);
int  width     = builder.getScalarTypeWidth(typeId);
spv::Id scalar = builder.getScalarTypeId(typeId);
spv::Id contained = builder.getContainedTypeId(typeId);      // single member
spv::Id contained = builder.getContainedTypeId(typeId, n);   // nth member
unsigned cols = builder.getNumColumns(id);
unsigned rows = builder.getNumRows(id);
unsigned comps = builder.getNumComponents(id);
```

## Constants

Created constants are deduplicated (except specialization constants).

```cpp
spv::Id trueConst  = builder.makeBoolConstant(true);
spv::Id falseConst = builder.makeBoolConstant(false);

spv::Id i32_5  = builder.makeIntConstant(5);
spv::Id u32_7  = builder.makeUintConstant(7);
spv::Id i64_9  = builder.makeInt64Constant(9);
spv::Id u64_11 = builder.makeUint64Constant(11);
spv::Id i8_1   = builder.makeInt8Constant(1);
spv::Id u8_2   = builder.makeUint8Constant(2);
spv::Id i16_3  = builder.makeInt16Constant(3);
spv::Id u16_4  = builder.makeUint16Constant(4);

spv::Id f32_1  = builder.makeFloatConstant(1.0f);
spv::Id f64_2  = builder.makeDoubleConstant(2.0);
spv::Id f16_3  = builder.makeFloat16Constant(3.0f);
spv::Id bf16_4 = builder.makeBFloat16Constant(4.0f);

spv::Id fpVal  = builder.makeFpConstant(floatTy, 1.5, false);
spv::Id nullVal = builder.makeNullConstant(structTy);

// Composite constant
std::vector<spv::Id> comps = { f32_1, f32_1, f32_1, f32_1 };
spv::Id vec4Const = builder.makeCompositeConstant(vec4Ty, comps);

// Specialization constants
spv::Id specI32 = builder.makeIntConstant(builder.makeIntType(32), 10, true);  // specConstant=true
spv::Id specVec = builder.makeCompositeConstant(vec4Ty, comps, true);
```

## Variables

```cpp
// Global variable
spv::Id globalVar = builder.createVariable(
    spv::Decoration::NoPrecision,
    spv::StorageClass::Private,
    floatTy,
    "myGlobal",
    builder.makeFloatConstant(0.0f)
);

// Local variable (Function storage class)
spv::Id localVar = builder.createVariable(
    spv::Decoration::NoPrecision,
    spv::StorageClass::Function,
    floatTy,
    "myLocal"
);

// Untyped variable
spv::Id untypedVar = builder.createUntypedVariable(
    spv::Decoration::NoPrecision,
    spv::StorageClass::StorageBuffer,
    "untypedVar",
    dataTypeId,   // optional data type
    initializerId // optional initializer
);

// Undefined value
spv::Id undef = builder.createUndefined(floatTy);
```

## Functions

```cpp
// Entry point
spv::Function* entry = builder.makeEntryPoint("main");
builder.addEntryPoint(spv::ExecutionModel::Fragment, entry, "main");
builder.addExecutionMode(entry, spv::ExecutionMode::OriginUpperLeft);

// Regular function
Block* entryBlock = nullptr;
std::vector<spv::Id> paramTypes = { floatTy, int32Ty };
std::vector<std::vector<spv::Decoration>> paramDecs = {
    { spv::Decoration::NoPrecision },
    { spv::Decoration::NoPrecision }
};
spv::Function* func = builder.makeFunctionEntry(
    spv::Decoration::NoPrecision,
    floatTy,           // return type
    "myFunc",
    spv::LinkageType::Max,
    paramTypes,
    paramDecs,
    &entryBlock
);

builder.enterFunction(func);
builder.setBuildPoint(entryBlock);

// Parameters are assigned IDs sequentially from a base ID:
spv::Id p0 = func->getParamId(0);
spv::Id p1 = func->getParamId(1);

// Body instructions...
builder.makeReturn(false, resultId);

builder.leaveFunction();
```

## Control Flow

### If-Then-Else

```cpp
spv::Id cond = ...; // bool
{
    spv::Builder::If ifBuilder(cond, spv::SelectionControlMask::MaskNone, builder);
    // then block
    builder.createStore(..., ...);

    ifBuilder.makeBeginElse();
    // else block
    builder.createStore(..., ...);

    ifBuilder.makeEndIf();
}
// builder is now at merge block
```

### Switch

```cpp
std::vector<int> caseValues = { 0, 1 };
std::vector<int> valueToSegment = { 0, 1 };
int defaultSegment = 2;
int numSegments = 3;
std::vector<Block*> segmentBB;

builder.makeSwitch(selectorId, spv::SelectionControlMask::MaskNone,
                   numSegments, caseValues, valueToSegment, defaultSegment, segmentBB);

// Segment 0 (case 0)
builder.nextSwitchSegment(segmentBB, 0);
// ... code ...
builder.addSwitchBreak(false);

// Segment 1 (case 1)
builder.nextSwitchSegment(segmentBB, 1);
// ... code ...
builder.addSwitchBreak(false);

// Segment 2 (default)
builder.nextSwitchSegment(segmentBB, 2);
// ... code ...
builder.addSwitchBreak(false);

builder.endSwitch(segmentBB);
```

### Loops

```cpp
spv::Builder::LoopBlocks& loop = builder.makeNewLoop();
builder.setBuildPoint(&loop.head);
// loop header test
builder.createBranch(false, &loop.body);

builder.setBuildPoint(&loop.body);
// loop body
// continue
builder.createLoopContinue();

builder.setBuildPoint(&loop.continue_target);
// continue block (e.g., increment)
builder.createBranch(false, &loop.head);

builder.setBuildPoint(&loop.merge);
// after loop

builder.closeLoop();

// Break / continue inside loop
builder.createLoopExit();     // break
builder.createLoopContinue(); // continue
```

## Arithmetic & Logic Instructions

```cpp
// Unary
spv::Id neg = builder.createUnaryOp(spv::Op::OpSNegate, int32Ty, val);
spv::Id notb = builder.createUnaryOp(spv::Op::OpLogicalNot, boolTy, bval);

// Binary
spv::Id add = builder.createBinOp(spv::Op::OpFAdd, floatTy, a, b);
spv::Id sub = builder.createBinOp(spv::Op::OpISub, int32Ty, a, b);
spv::Id mul = builder.createBinOp(spv::Op::OpIMul, int32Ty, a, b);
spv::Id div = builder.createBinOp(spv::Op::OpFDiv, floatTy, a, b);
spv::Id and_ = builder.createBinOp(spv::Op::OpBitwiseAnd, uint32Ty, a, b);

// Ternary
spv::Id fma = builder.createTriOp(spv::Op::OpExtInst, floatTy, glsl450, GLSLstd450Fma, a, b, c);

// Generic n-ary
std::vector<spv::Id> ops = { a, b, c };
spv::Id res = builder.createOp(spv::Op::OpVectorTimesMatrix, vec4Ty, ops);

// Generic with immediates mixed
std::vector<spv::IdImmediate> mixed;
mixed.emplace_back(true, idOp);
mixed.emplace_back(false, (unsigned)spv::MemoryAccessMask::Aligned);
spv::Id r = builder.createOp(spv::Op::Op..., typeId, mixed);

// SpecConstantOp
spv::Id specAdd = builder.createSpecConstantOp(
    spv::Op::OpIAdd, int32Ty,
    { specA, specB },
    {} // literals
);
```

## Memory Instructions

```cpp
// Load / Store
spv::Id loaded = builder.createLoad(ptrId, spv::Decoration::NoPrecision);
builder.createStore(valueId, ptrId);

// With memory access
builder.createStore(valueId, ptrId,
    spv::MemoryAccessMask::NonUniformPointerEXT,
    spv::Scope::Device, 4);

// Access chain
std::vector<spv::Id> indexes = { builder.makeUintConstant(0), builder.makeUintConstant(2) };
spv::Id chain = builder.createAccessChain(spv::StorageClass::Function, basePtr, indexes);

// Array length
spv::Id len = builder.createArrayLength(bufferPtr, memberIndex, 32);

// Composite extract / insert
spv::Id elem = builder.createCompositeExtract(composite, elemType, 2);
spv::Id elem = builder.createCompositeExtract(composite, elemType, std::vector<unsigned>{0, 1});
spv::Id inserted = builder.createCompositeInsert(newVal, composite, compositeType, 0);

// Vector dynamic extract / insert
spv::Id dynElem = builder.createVectorExtractDynamic(vec, elemType, indexId);
spv::Id dynVec = builder.createVectorInsertDynamic(vec, vecType, newElem, indexId);

// Composite construct
std::vector<spv::Id> constituents = { a, b, c, d };
spv::Id comp = builder.createCompositeConstruct(vec4Ty, constituents);

// Constructors (vector/matrix/scalar promotion)
std::vector<spv::Id> srcs = { scalarId };
spv::Id vec4Id = builder.createConstructor(spv::Decoration::NoPrecision, srcs, vec4Ty);
spv::Id matId = builder.createMatrixConstructor(spv::Decoration::NoPrecision, srcs, mat4x4Ty);

// Swizzle
std::vector<unsigned> channels = { 2, 1, 0, 3 };
spv::Id swizzled = builder.createRvalueSwizzle(spv::Decoration::NoPrecision, vec4Ty, vec, channels);
spv::Id merged  = builder.createLvalueSwizzle(vec4Ty, target, source, channels);

// Scalar promotion (smear scalar to vector width)
Id left = scalar, right = vector;
builder.promoteScalar(spv::Decoration::NoPrecision, left, right);
// one of the arguments is rewritten in-place

// Smear explicitly
spv::Id smeared = builder.smearScalar(spv::Decoration::NoPrecision, scalarId, vec4Ty);
```

## Access Chain Helper

`Builder` maintains one active access chain for l-value / r-value tracking.

```cpp
builder.clearAccessChain();
builder.setAccessChainLValue(ptrId);     // base is a pointer
builder.setAccessChainRValue(valueId);   // base is an r-value

builder.accessChainPush(indexId, coherentFlags, alignment);
builder.accessChainPushSwizzle(swizzleChannels, preSwizzleBaseType, coherentFlags, alignment);
builder.accessChainPushComponent(componentId, preSwizzleBaseType, coherentFlags, alignment);

// Load through the chain
spv::Id result = builder.accessChainLoad(
    spv::Decoration::NoPrecision,   // precision
    spv::Decoration::Max,           // l-value nonuniform
    spv::Decoration::Max,           // r-value nonuniform
    resultType,
    spv::MemoryAccessMask::MaskNone,
    spv::Scope::Max, 0
);

// Store through the chain
builder.accessChainStore(valueId, spv::Decoration::Max);

// Get direct l-value pointer
spv::Id lval = builder.accessChainGetLValue();

// Get inferred type after chain dereferences
spv::Id inferred = builder.accessChainGetInferredType();

// Save/restore chain
spv::Builder::AccessChain saved = builder.getAccessChain();
builder.setAccessChain(saved);
```

## Texture Operations

```cpp
spv::Builder::TextureParameters params;
params.sampler = sampledImageId;
params.coords = coordsId;
params.bias = spv::NoResult;
params.lod = lodId;
params.Dref = spv::NoResult;
params.offset = offsetId;
params.gradX = spv::NoResult;
params.gradY = spv::NoResult;
params.component = spv::NoResult;
params.nonprivate = false;
params.volatil = false;

spv::Id texResult = builder.createTextureCall(
    spv::Decoration::NoPrecision,
    resultType,
    false,   // sparse
    false,   // fetch
    false,   // proj
    false,   // gather
    false,   // noImplicit
    params,
    spv::ImageOperandsMask::MaskNone
);
```

## Decorations & Names

```cpp
builder.addName(id, "myVar");
builder.addMemberName(structTy, 0, "field0");

builder.addDecoration(id, spv::Decoration::Location, 0);
builder.addDecoration(id, spv::Decoration::Binding, 2);
builder.addDecoration(id, spv::Decoration::DescriptorSet, 0);
builder.addDecoration(id, spv::Decoration::NoContraction);
builder.addDecoration(id, spv::Decoration::RelaxedPrecision);
builder.addDecoration(id, spv::Decoration::BuiltIn, (int)spv::BuiltIn::Position);

// Member decorations
builder.addMemberDecoration(structTy, 0, spv::Decoration::Offset, 0);
builder.addMemberDecoration(structTy, 1, spv::Decoration::Offset, 16);

// Decoration with vector literals
builder.addDecoration(id, spv::Decoration::WorkgroupSize,
    std::vector<unsigned>{64, 1, 1});

// DecorationId
builder.addDecorationId(id, spv::Decoration::ArrayStrideIdEXT, strideId);

// Linkage
builder.addLinkageDecoration(id, "myFunc", spv::LinkageType::Export);
```

## Barriers

```cpp
builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device,
    spv::MemorySemanticsMask::UniformMemory |
    spv::MemorySemanticsMask::WorkgroupMemory);

builder.createMemoryBarrier(spv::Scope::Device,
    spv::MemorySemanticsMask::ImageMemory);
```

## Debug Info

### SPIR-V Standard Debug (OpLine / OpSource)

```cpp
builder.setEmitSpirvDebugInfo();
builder.setDebugMainSourceFile("shader.glsl");
builder.setDebugSourceLocation(42, "shader.glsl");
builder.setSourceText(sourceText);
```

### NonSemantic Shader Debug Info

```cpp
builder.setEmitNonSemanticShaderDebugInfo(true); // true = emit source text

// Types are automatically mapped. Retrieve debug type:
spv::Id debugType = builder.getDebugType(spirvTypeId);

// Debug scopes / lexical blocks
builder.enterLexicalBlock(line, column);
builder.leaveLexicalBlock();

// Debug function setup (call after makeFunctionEntry)
builder.setupFunctionDebugInfo(func, "myFunc", paramTypes, paramNames);

// Debug variables
spv::Id dbgGlobal = builder.createDebugGlobalVariable(debugType, "globalVar", varId);
spv::Id dbgLocal  = builder.createDebugLocalVariable(debugType, "localVar", argNumber);
spv::Id dbgDecl   = builder.makeDebugDeclare(dbgLocal, ptrId);
spv::Id dbgVal    = builder.makeDebugValue(dbgLocal, valueId);
```

## Function Calls

```cpp
spv::Id result = builder.createFunctionCall(calleeFunc, { arg0, arg1, arg2 });
```

## Builtin Call (GLSL.std.450)

```cpp
spv::Id glsl450 = builder.import("GLSL.std.450");
spv::Id sqrtVal = builder.createBuiltinCall(floatTy, glsl450, GLSLstd450Sqrt, { val });
```

## Post-Processing

```cpp
builder.postProcess(false);      // prune unreachable blocks, add caps/extensions
builder.postProcessCFG();        // prune unreachable blocks
builder.postProcessFeatures();   // add capabilities/extensions from instructions
builder.postProcessSamplers();   // move OpSampledImage next to users
```

## Serialization

```cpp
std::vector<unsigned int> spirv;
builder.dump(spirv);

// Disassemble
spv::Disassemble(std::cout, spirv);

// Save binary
glslang::OutputSpvBin(spirv, "out.spv");

// Save C header
glslang::OutputSpvHex(spirv, "out.h", "g_spv");
```

## IR Classes

`spvIR.h` defines the in-memory IR:

```cpp
// Instruction
spv::Instruction* inst = new spv::Instruction(resultId, typeId, spv::Op::OpIAdd);
inst->addIdOperand(opA);
inst->addIdOperand(opB);

// Block (owns instructions)
spv::Block* block = new spv::Block(blockId, function);
block->addInstruction(std::unique_ptr<spv::Instruction>(inst));
block->addLocalVariable(std::unique_ptr<spv::Instruction>(varInst));
bool terminated = block->isTerminated();

// Function (owns blocks)
spv::Function* func = new spv::Function(funcId, retType, funcType, firstParamId,
                                        linkage, name, module);
func->addBlock(block);
func->setReturnPrecision(spv::Decoration::RelaxedPrecision);
func->addParamPrecision(0, spv::Decoration::RelaxedPrecision);

// Module (owns functions, instruction map)
spv::Module module;
module.addFunction(func);
module.mapInstruction(inst);
spv::Instruction* found = module.getInstruction(id);
spv::Id typeId = module.getTypeId(resultId);
```

## GlslangToSpv Bridge

Convert glslang intermediate representation to SPIR-V using the builder internally:

```cpp
#include "SPIRV/GlslangToSpv.h"

glslang::TIntermediate* im = program.getIntermediate(EShLangVertex);
std::vector<unsigned int> spirv;
spv::SpvBuildLogger logger;
glslang::SpvOptions options;
options.generateDebugInfo = false;
options.disableOptimizer = true;
options.validate = true;

glslang::GlslangToSpv(*im, spirv, &logger, &options);
// logger.getAllMessages()
```

## Key Types Summary

| Type | Purpose |
|------|---------|
| `spv::Builder` | SPIR-V module construction |
| `spv::Instruction` | Single SPIR-V instruction |
| `spv::Block` | Basic block (label + instructions) |
| `spv::Function` | SPIR-V function |
| `spv::Module` | Module root, ID-to-instruction map |
| `spv::Builder::If` | Structured if-then-else helper |
| `spv::Builder::LoopBlocks` | Structured loop blocks |
| `spv::Builder::AccessChain` | L-value / R-value access chain |
| `spv::Builder::TextureParameters` | Texture op parameters |
| `spv::IdImmediate` | Operand that is either ID or immediate |
| `glslang::SpvOptions` | Options for `GlslangToSpv` |
