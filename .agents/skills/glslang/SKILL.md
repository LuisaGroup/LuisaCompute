---
name: glslang
description: glslang SPIR-V builder API — types, constants, functions, control flow, memory, textures, decorations, debug info, and GlslangToSpv patterns
---

# glslang SPIR-V Usage

Located in `src/ext/glslang/SPIRV`. Headers:

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
// ... build ...
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
spv::Id glsl450 = builder.import("GLSL.std.450");
```

## Types (canonicalized)

```cpp
spv::Id voidTy   = builder.makeVoidType();
spv::Id boolTy   = builder.makeBoolType();
spv::Id int32Ty  = builder.makeIntType(32);
spv::Id uint32Ty = builder.makeUintType(32);
spv::Id uint64Ty = builder.makeUintType(64);
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

std::vector<spv::Id> members = {floatTy, int32Ty};
spv::Id structTy = builder.makeStructType(members, {}, "MyStruct", false);

spv::Id ptrTy     = builder.makePointer(spv::StorageClass::Function, floatTy);
spv::Id fwdPtrTy  = builder.makeForwardPointer(spv::StorageClass::PhysicalStorageBufferEXT);
spv::Id untypedPtr= builder.makeUntypedPointer(spv::StorageClass::StorageBuffer);
spv::Id fnTy      = builder.makeFunctionType(voidTy, {floatTy, int32Ty});

spv::Id imgTy       = builder.makeImageType(floatTy, spv::Dim::Dim2D, false, false, false, 1, spv::ImageFormat::Rgba32f, "texture2D");
spv::Id sampledImgTy= builder.makeSampledImageType(imgTy, "sampler2D");
spv::Id samplerTy   = builder.makeSamplerType("sampler");

spv::Id asTy = builder.makeAccelerationStructureType();
spv::Id rqTy = builder.makeRayQueryType();
spv::Id hoTy = builder.makeHitObjectEXTType();

spv::Id coopMatTy = builder.makeCooperativeMatrixTypeKHR(floatTy, scopeId, rowsId, colsId, useId);
spv::Id coopVecTy = builder.makeCooperativeVectorTypeNV(floatTy, componentsId);
spv::Id tensorTy  = builder.makeTensorTypeARM(floatTy, rankId);

// Generic
std::vector<spv::IdImmediate> ops = {{true, someId}};
spv::Id genericTy = builder.makeGenericType(spv::Op::OpType..., ops);
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
spv::Id contained = builder.getContainedTypeId(typeId);     // single
spv::Id contained = builder.getContainedTypeId(typeId, n);  // nth
unsigned cols = builder.getNumColumns(id);
unsigned rows = builder.getNumRows(id);
unsigned comps= builder.getNumComponents(id);
```

## Constants (deduplicated; spec constants not)

```cpp
spv::Id t = builder.makeBoolConstant(true), f = builder.makeBoolConstant(false);
spv::Id i32 = builder.makeIntConstant(5), u32 = builder.makeUintConstant(7);
spv::Id i64 = builder.makeInt64Constant(9), u64 = builder.makeUint64Constant(11);
spv::Id i8  = builder.makeInt8Constant(1),  u8  = builder.makeUint8Constant(2);
spv::Id i16 = builder.makeInt16Constant(3), u16 = builder.makeUint16Constant(4);
spv::Id f32 = builder.makeFloatConstant(1.0f), f64 = builder.makeDoubleConstant(2.0);
spv::Id f16 = builder.makeFloat16Constant(3.0f), bf16 = builder.makeBFloat16Constant(4.0f);
spv::Id fp  = builder.makeFpConstant(floatTy, 1.5, false);
spv::Id null= builder.makeNullConstant(structTy);

// Composite
spv::Id vec4 = builder.makeCompositeConstant(vec4Ty, {f32, f32, f32, f32});

// Spec constants
spv::Id specI32 = builder.makeIntConstant(builder.makeIntType(32), 10, true);
spv::Id specVec = builder.makeCompositeConstant(vec4Ty, comps, true);
```

## Variables

```cpp
spv::Id global = builder.createVariable(spv::Decoration::NoPrecision, spv::StorageClass::Private, floatTy, "g", builder.makeFloatConstant(0.0f));
spv::Id local  = builder.createVariable(spv::Decoration::NoPrecision, spv::StorageClass::Function, floatTy, "l");
spv::Id untyped= builder.createUntypedVariable(spv::Decoration::NoPrecision, spv::StorageClass::StorageBuffer, "u", dataTypeId, initId);
spv::Id undef  = builder.createUndefined(floatTy);
```

## Functions

```cpp
// Entry point
spv::Function* entry = builder.makeEntryPoint("main");
builder.addEntryPoint(spv::ExecutionModel::Fragment, entry, "main");
builder.addExecutionMode(entry, spv::ExecutionMode::OriginUpperLeft);

// Regular function
spv::Block* entryBlock = nullptr;
spv::Function* func = builder.makeFunctionEntry(
    spv::Decoration::NoPrecision, floatTy, "myFunc", spv::LinkageType::Max,
    {floatTy, int32Ty},
    {{spv::Decoration::NoPrecision}, {spv::Decoration::NoPrecision}},
    &entryBlock);

builder.enterFunction(func);
builder.setBuildPoint(entryBlock);
spv::Id p0 = func->getParamId(0);
spv::Id p1 = func->getParamId(1);
builder.makeReturn(false, resultId);  // or makeReturn(false) for void
builder.leaveFunction();
```

## Control Flow

### If-Then-Else
```cpp
spv::Builder::If ifBuilder(cond, spv::SelectionControlMask::MaskNone, builder);
// then block
ifBuilder.makeBeginElse();
// else block
ifBuilder.makeEndIf();
// merge block
```

### Switch
```cpp
std::vector<int> caseValues = {0, 1}, valueToSegment = {0, 1};
int defaultSegment = 2, numSegments = 3;
std::vector<Block*> segmentBB;
builder.makeSwitch(selectorId, spv::SelectionControlMask::MaskNone, numSegments, caseValues, valueToSegment, defaultSegment, segmentBB);
builder.nextSwitchSegment(segmentBB, 0); /* ... */ builder.addSwitchBreak(false);
builder.nextSwitchSegment(segmentBB, 1); /* ... */ builder.addSwitchBreak(false);
builder.nextSwitchSegment(segmentBB, 2); /* ... */ builder.addSwitchBreak(false);
builder.endSwitch(segmentBB);
```

### Loops
```cpp
spv::Builder::LoopBlocks& loop = builder.makeNewLoop();
builder.setBuildPoint(&loop.head);
builder.createBranch(false, &loop.body);
builder.setBuildPoint(&loop.body);
// loop body
builder.createLoopContinue();
builder.setBuildPoint(&loop.continue_target);
builder.createBranch(false, &loop.head);
builder.setBuildPoint(&loop.merge);
builder.closeLoop();
// break: builder.createLoopExit();  continue: builder.createLoopContinue();
```

## Arithmetic & Logic

```cpp
spv::Id neg  = builder.createUnaryOp(spv::Op::OpSNegate, int32Ty, val);
spv::Id notb = builder.createUnaryOp(spv::Op::OpLogicalNot, boolTy, bval);
spv::Id add  = builder.createBinOp(spv::Op::OpFAdd, floatTy, a, b);
spv::Id sub  = builder.createBinOp(spv::Op::OpISub, int32Ty, a, b);
spv::Id mul  = builder.createBinOp(spv::Op::OpIMul, int32Ty, a, b);
spv::Id div  = builder.createBinOp(spv::Op::OpFDiv, floatTy, a, b);
spv::Id and_ = builder.createBinOp(spv::Op::OpBitwiseAnd, uint32Ty, a, b);

// ExtInst (ternary)
spv::Id fma = builder.createOp(spv::Op::OpExtInst, floatTy, {glsl450, GLSLstd450Fma, a, b, c});

// Generic n-ary
spv::Id r = builder.createOp(spv::Op::OpVectorTimesMatrix, vec4Ty, {a, b, c});

// Mixed ID/immediates
std::vector<spv::IdImmediate> mixed = {{true, idOp}, {false, (unsigned)spv::MemoryAccessMask::Aligned}};
spv::Id r = builder.createOp(spv::Op::Op..., typeId, mixed);

// SpecConstantOp
spv::Id specAdd = builder.createSpecConstantOp(spv::Op::OpIAdd, int32Ty, {specA, specB}, {});
```

## Memory Instructions

```cpp
spv::Id loaded = builder.createLoad(ptrId, spv::Decoration::NoPrecision);
builder.createStore(valueId, ptrId);
builder.createStore(valueId, ptrId, spv::MemoryAccessMask::NonUniformPointerEXT, spv::Scope::Device, 4);

// Access chain
std::vector<spv::Id> indexes = {builder.makeUintConstant(0), builder.makeUintConstant(2)};
spv::Id chain = builder.createAccessChain(spv::StorageClass::Function, basePtr, indexes);

// Composite
spv::Id elem  = builder.createCompositeExtract(composite, elemType, 2);
spv::Id elem  = builder.createCompositeExtract(composite, elemType, std::vector<unsigned>{0, 1});
spv::Id ins   = builder.createCompositeInsert(newVal, composite, compositeType, 0);
spv::Id dynEl = builder.createVectorExtractDynamic(vec, elemType, indexId);
spv::Id dynVec= builder.createVectorInsertDynamic(vec, vecType, newElem, indexId);
spv::Id comp  = builder.createCompositeConstruct(vec4Ty, {a, b, c, d});

spv::Id vec4 = builder.createConstructor(spv::Decoration::NoPrecision, {scalarId}, vec4Ty);
spv::Id mat  = builder.createMatrixConstructor(spv::Decoration::NoPrecision, srcs, mat4x4Ty);

// Swizzle
spv::Id swz = builder.createRvalueSwizzle(spv::Decoration::NoPrecision, vec4Ty, vec, {2, 1, 0, 3});
spv::Id lswz= builder.createLvalueSwizzle(vec4Ty, target, source, {2, 1, 0, 3});

// Scalar promotion (in-place)
builder.promoteScalar(spv::Decoration::NoPrecision, left, right);
spv::Id smeared = builder.smearScalar(spv::Decoration::NoPrecision, scalarId, vec4Ty);
```

## Access Chain Helper

Builder maintains one active access chain for l-value/r-value tracking:

```cpp
builder.clearAccessChain();
builder.setAccessChainLValue(ptrId);    // base is pointer
builder.setAccessChainRValue(valueId);  // base is r-value
builder.accessChainPush(indexId, coherentFlags, alignment);
builder.accessChainPushSwizzle(channels, preSwizzleBaseType, coherentFlags, alignment);
builder.accessChainPushComponent(componentId, preSwizzleBaseType, coherentFlags, alignment);

spv::Id result = builder.accessChainLoad(precision, lvalNonUniform, rvalNonUniform, resultType, memAccess, scope, n);
builder.accessChainStore(valueId, nonUniform);
spv::Id lval = builder.accessChainGetLValue();
spv::Id inferred = builder.accessChainGetInferredType();

// Save/restore
spv::Builder::AccessChain saved = builder.getAccessChain();
builder.setAccessChain(saved);
```

## Texture Operations

```cpp
spv::Builder::TextureParameters params = {};
params.sampler = sampledImageId;
params.coords = coordsId;
params.lod = lodId;  // etc: bias, Dref, offset, gradX, gradY, component, sample, lodClamp, ...
// nonprivate, volatil, nontemporal = false

spv::Id tex = builder.createTextureCall(precision, resultType,
    false/*sparse*/, false/*fetch*/, false/*proj*/, false/*gather*/, false/*noImplicit*/,
    params, spv::ImageOperandsMask::MaskNone);
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
builder.addMemberDecoration(structTy, 0, spv::Decoration::Offset, 0);
builder.addMemberDecoration(structTy, 1, spv::Decoration::Offset, 16);
builder.addDecoration(id, spv::Decoration::WorkgroupSize, std::vector<unsigned>{64, 1, 1});
builder.addDecorationId(id, spv::Decoration::ArrayStrideIdEXT, strideId);
builder.addLinkageDecoration(id, "myFunc", spv::LinkageType::Export);
```

## Barriers

```cpp
builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device,
    spv::MemorySemanticsMask::UniformMemory | spv::MemorySemanticsMask::WorkgroupMemory);
builder.createMemoryBarrier(spv::Scope::Device, spv::MemorySemanticsMask::ImageMemory);
```

## Debug Info

### SPIR-V Standard (OpLine/OpSource)
```cpp
builder.setEmitSpirvDebugInfo();
builder.setDebugMainSourceFile("shader.glsl");
builder.setDebugSourceLocation(42, "shader.glsl");
builder.setSourceText(sourceText);
```

### NonSemantic Shader Debug Info
```cpp
builder.setEmitNonSemanticShaderDebugInfo(true);
spv::Id debugType = builder.getDebugType(spirvTypeId);
builder.enterLexicalBlock(line, column);
builder.leaveLexicalBlock();
builder.setupFunctionDebugInfo(func, "myFunc", paramTypes, paramNames);
spv::Id dbgGlobal = builder.createDebugGlobalVariable(debugType, "globalVar", varId);
spv::Id dbgLocal  = builder.createDebugLocalVariable(debugType, "localVar", argNumber);
spv::Id dbgDecl   = builder.makeDebugDeclare(dbgLocal, ptrId);
spv::Id dbgVal    = builder.makeDebugValue(dbgLocal, valueId);
```

## Function Calls & Builtins

```cpp
spv::Id result = builder.createFunctionCall(calleeFunc, {arg0, arg1, arg2});
spv::Id sqrtVal = builder.createBuiltinCall(floatTy, glsl450, GLSLstd450Sqrt, {val});
```

## Post-Processing & Serialization

```cpp
builder.postProcess(false);       // prune + caps/extensions
builder.postProcessCFG();         // prune unreachable
builder.postProcessFeatures();    // add caps/extensions from instructions
builder.postProcessSamplers();    // move OpSampledImage near users

std::vector<unsigned int> spirv;
builder.dump(spirv);
spv::Disassemble(std::cout, spirv);
glslang::OutputSpvBin(spirv, "out.spv");
glslang::OutputSpvHex(spirv, "out.h", "g_spv");
```

## IR Classes (`spvIR.h`)

```cpp
spv::Instruction* inst = new spv::Instruction(resultId, typeId, spv::Op::OpIAdd);
inst->addIdOperand(opA);
inst->addIdOperand(opB);

spv::Block* block = new spv::Block(blockId, *function);
block->addInstruction(std::unique_ptr<spv::Instruction>(inst));
block->addLocalVariable(std::unique_ptr<spv::Instruction>(varInst));
bool terminated = block->isTerminated();

spv::Function* func = new spv::Function(funcId, retType, funcType, firstParamId, linkage, name, module);
func->addBlock(block);
func->setReturnPrecision(spv::Decoration::RelaxedPrecision);
func->addParamPrecision(0, spv::Decoration::RelaxedPrecision);

spv::Module module;
module.addFunction(func);
module.mapInstruction(inst);
spv::Instruction* found = module.getInstruction(id);
spv::Id typeId = module.getTypeId(resultId);
```

## Key Types

| Type | Purpose |
|---|---|
| `spv::Builder` | SPIR-V module construction |
| `spv::Instruction` | Single SPIR-V instruction |
| `spv::Block` | Basic block |
| `spv::Function` | SPIR-V function |
| `spv::Module` | Module root, ID→instruction map |
| `spv::Builder::If` | Structured if-then-else helper |
| `spv::Builder::LoopBlocks` | Structured loop blocks |
| `spv::Builder::AccessChain` | L-value/R-value access chain |
| `spv::Builder::TextureParameters` | Texture op parameters |
| `spv::IdImmediate` | Operand: ID or immediate |
| `glslang::SpvOptions` | GlslangToSpv options |

## GlslangToSpv Patterns

From `TGlslangToSpvTraverser` (`src/ext/glslang/SPIRV/GlslangToSpv.cpp`). Common pattern: clear access chain → traverse → load/store → set R-value.

### visitSymbol
```cpp
builder.clearAccessChain();
if (isRValue || !builder.isPointerType(builder.getTypeId(id)))
    builder.setAccessChainRValue(id);
else
    builder.setAccessChainLValue(id);

spv::StorageClass sc = builder.getStorageClass(id);
if (builder.isGlobalVariable(id))
    iOSet.insert(id);

builder.addExtension("SPV_GOOGLE_hlsl_functionality1");
builder.addDecorationId(id, spv::Decoration::HlslCounterBufferGOOGLE, counterId);
```

### visitBinary (Assignment)
```cpp
builder.clearAccessChain(); node->getLeft()->traverse(this);
auto lValue = builder.getAccessChain();
builder.clearAccessChain(); node->getRight()->traverse(this);
spv::Id rValue = accessChainLoad(node->getRight()->getType());
builder.setAccessChain(lValue);
multiTypeStore(node->getLeft()->getType(), rValue);
builder.clearAccessChain(); builder.setAccessChainRValue(rValue);
```

### visitBinary (Array/Vector Index)
```cpp
// zero-extend narrow uint indexes to 32-bit
if (builder.isUintType(indexType) && builder.getScalarTypeWidth(indexType) < 32)
    index = builder.createUnaryOp(spv::Op::OpUConvert, builder.makeUintType(32), index);
builder.accessChainPush(index, coherentFlags, alignment);
```

### visitBinary (Swizzle)
```cpp
builder.accessChainPushSwizzle(swizzle, convertGlslangToSpvType(node->getLeft()->getType()),
                               coherentFlags, alignment);
```

### visitUnary (Inc/Dec)
```cpp
spv::Id operand = builder.accessChainGetLValue();
spv::Id one = builder.makeIntConstant(1);
spv::Id result = builder.createBinOp(op, type, operand, one);
builder.accessChainStore(result, ...);
builder.clearAccessChain(); builder.setAccessChainRValue(result);
```

### visitUnary (Builtin / NoResult / ArrayLength)
```cpp
// Builtin
spv::Id result = builder.createBuiltinCall(resultType(), glsl450, opcode, {operand});
// No-result
builder.createNoResultOp(spv::Op::OpKill);
builder.createNoResultOp(spv::Op::OpTerminateInvocation);
builder.createNoResultOp(spv::Op::OpDemoteToHelperInvocationEXT);
builder.createNoResultOp(spv::Op::OpAssumeTrueKHR, operand);
// Array length
spv::Id len = builder.createArrayLength(builder.accessChainGetLValue(), member, bits);
len = builder.createUnaryOp(spv::Op::OpBitcast, builder.makeIntType(bits), len);
// Cooperative matrix/vector
spv::Id lenKHR = builder.createCooperativeMatrixLengthKHR(typeId);
spv::Id lenNV  = builder.createCooperativeMatrixLengthNV(typeId);
spv::Id lenVec = builder.getCooperativeVectorNumComponents(typeId);
// Tensor
spv::Id layout = builder.createOp(spv::Op::OpCreateTensorLayoutNV, resultType(), {});
spv::Id view   = builder.createOp(spv::Op::OpCreateTensorViewNV, resultType(), {});
```

### visitAggregate
```cpp
// Function entry/leave
builder.setBuildPoint(shaderEntry->getLastBlock());
builder.enterFunction(shaderEntry); /* body */ builder.leaveFunction();
// Function call
spv::Id result = builder.createFunctionCall(callee, arguments);
// Constructors
spv::Id c = builder.createConstructor(precision, arguments, resultType());
spv::Id m = builder.createMatrixConstructor(precision, arguments, resultType());
// Builtin
spv::Id r = builder.createBuiltinCall(resultType(), extInst, opcode, arguments);
// Texture
spv::Builder::TextureParameters params = {sampledImageId, coordsId, /*...*/};
spv::Id tex = builder.createTextureCall(precision, resultType(), sparse, fetch, proj, gather, noImplicit, params, mask);
// Sampled image
spv::Id sampled = builder.createOp(spv::Op::OpSampledImage, resultType(), {imageId, samplerId});
// Cooperative matrix conversion
spv::Id coop = builder.createCooperativeMatrixConversion(resultType(), arguments[0]);
// Variable
spv::Id var = builder.createVariable(precision, spv::StorageClass::Function, type, name, init);
// Load/store
spv::Id loaded = builder.createLoad(ptrId, precision);
builder.createStore(valueId, ptrId);
// Debug scopes
builder.enterLexicalBlock(loc.line, loc.column); /* body */ builder.leaveLexicalBlock();
```

### visitSelection
```cpp
// Scalar ternary
spv::Id result = builder.createTriOp(spv::Op::OpSelect, resultType(), cond, trueVal, falseVal);
// Vector selection via smeared scalar cond
spv::Id condVec = builder.smearScalar(precision, cond, builder.makeVectorType(boolTy, components));
spv::Id result  = builder.createUnaryOp(spv::Op::OpCopyLogical, resultType(), ...);
```

### visitSwitch
```cpp
std::vector<int> caseValues = {0,1,2}, valueToSegment = {0,1,2};
builder.makeSwitch(selectorId, spv::SelectionControlMask::MaskNone, 4, caseValues, valueToSegment, 3, segmentBB);
builder.nextSwitchSegment(segmentBB, 0); /* case 0 */ builder.addSwitchBreak(false);
// ...
builder.endSwitch(segmentBB);
```

### visitLoop
```cpp
spv::Builder::LoopBlocks& loop = builder.makeNewLoop();
builder.setBuildPoint(&loop.head);
builder.createLoopMerge(&loop.merge, &loop.continue_target, spv::LoopControlMask::MaskNone, {});
builder.createConditionalBranch(cond, &loop.body, &loop.merge);
builder.setBuildPoint(&loop.body); /* body */ builder.createLoopContinue();
builder.setBuildPoint(&loop.continue_target); /* increment */ builder.createBranch(false, &loop.head);
builder.setBuildPoint(&loop.merge);
builder.closeLoop();
```

### visitBranch
```cpp
builder.makeReturn(false, returnValue);   // with value
builder.makeReturn(false);                // void
builder.createLoopExit();                 // break
builder.createLoopContinue();             // continue
builder.makeStatementTerminator(spv::Op::OpKill, "post-discard");
builder.makeStatementTerminator(spv::Op::OpTerminateInvocation, "post-terminate");
builder.createNoResultOp(spv::Op::OpDemoteToHelperInvocationEXT);
builder.makeStatementTerminator(spv::Op::OpTerminateRayKHR, "post-terminate-ray");
builder.makeStatementTerminator(spv::Op::OpIgnoreIntersectionKHR, "post-ignore");
```

### visitConstantUnion
```cpp
spv::Id constantId = createSpvConstant(node);
builder.clearAccessChain();
builder.setAccessChainRValue(constantId);
```

### visitVariableDecl
```cpp
builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
```

### visitFunctions
```cpp
// No direct builder usage; drives traversal of the translation unit.
```
