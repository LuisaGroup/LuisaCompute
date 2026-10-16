---
name: glslang
description: glslang SPIR-V Builder API for types, instructions, control flow, and decorations.
---

# glslang SPIR-V Usage

Located in `src/ext/glslang/SPIRV`. Headers:

```cpp
#include "SPIRV/SpvBuilder.h"
#include "SPIRV/spvIR.h"
#include "SPIRV/GlslangToSpv.h"
#include "SPIRV/disassemble.h"
```

Consumers in this repo: the native XIR->SPIR-V codegen owns one builder per module
(`luisa::make_unique<spv::Builder>(spv::Spv_1_5, 0, &_logger)` in
`src/backends/common/spirv/spirv_codegen/emit.cpp`, held as `_builder_ptr`/`_builder` in
`src/backends/common/spirv/spirv_codegen/entry.h`; emitted from `type.cpp`, `bind.cpp`,
`instruction.cpp`). The GLSL path in `src/backends/vk/glslang_compiler.cpp`
(`compile_glsl_to_spirv`, used by `native_shader_ext`) goes through
`glslang::GlslangToSpv(*intermediate, words, &options)` with `glslang::SpvOptions` instead
of driving `spv::Builder` directly.

> Code snippets follow glslang's own conventions (e.g. `camelCase` builder methods). LuisaCompute project style rules apply to project code, while `src/ext/glslang` is third-party code.

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
builder.setEmitSpirvDebugInfo();                    // required before setting debug locations
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
// Second argument is member debug info; use {} when no per-member debug data is needed.
spv::Id structTy = builder.makeStructType(members, {}, "MyStruct", false);

spv::Id ptrTy     = builder.makePointer(spv::StorageClass::Function, floatTy);
spv::Id fwdPtrTy  = builder.makeForwardPointer(spv::StorageClass::PhysicalStorageBuffer);
// Resolve a forward pointer to its pointee type once the pointee is known.
spv::Id resolvedPtrTy = builder.makePointerFromForwardPointer(spv::StorageClass::PhysicalStorageBuffer, fwdPtrTy, floatTy);
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

// Generic (opcode-driven type creation, e.g. tensor layout/view types)
std::vector<spv::IdImmediate> ops = {{true, rankId}, {true, dimId}};
spv::Id genericTy = builder.makeGenericType(spv::Op::OpTypeTensorLayoutNV, ops);
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
spv::Id specVec = builder.makeCompositeConstant(vec4Ty, {f32, f32, f32, f32}, true);
```

## Variables

```cpp
spv::Id global = builder.createVariable(spv::NoPrecision, spv::StorageClass::Private, floatTy, "g", builder.makeFloatConstant(0.0f));
spv::Id local  = builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, floatTy, "l");
spv::Id untyped= builder.createUntypedVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, "u", dataTypeId, initId);
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
    spv::NoPrecision, floatTy, "myFunc", spv::LinkageType::Max,
    {floatTy, int32Ty},
    {{spv::NoPrecision}, {spv::NoPrecision}},
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
builder.createLoopMerge(&loop.merge, &loop.continue_target, spv::LoopControlMask::MaskNone, {});
builder.createConditionalBranch(cond, &loop.body, &loop.merge);
builder.setBuildPoint(&loop.body);
// loop body
builder.createLoopContinue();
builder.setBuildPoint(&loop.continue_target);
// loop increment (optional)
builder.createBranch(true, &loop.head); // implicit back edge (no debug source location)
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

// OpExtInst: the entry point is a literal, so use createBuiltinCall (not createOp)
spv::Id fma = builder.createBuiltinCall(floatTy, glsl450, GLSLstd450Fma, {a, b, c});

// Generic operand-vector form (any createOp arity)
spv::Id r = builder.createOp(spv::Op::OpVectorTimesMatrix, vec4Ty, {vec, mat});

// Mixed ID/immediates (literal words mixed with IDs, e.g. a group-op reduction literal)
std::vector<spv::IdImmediate> mixed = {{true, scopeId},
                                       {false, (unsigned)spv::GroupOperation::Reduce},
                                       {true, val}};
spv::Id reduced = builder.createOp(spv::Op::OpGroupNonUniformBitwiseAnd, typeId, mixed);

// SpecConstantOp
spv::Id specAdd = builder.createSpecConstantOp(spv::Op::OpIAdd, int32Ty, {specA, specB}, {});
```

## Memory Instructions

```cpp
spv::Id loaded = builder.createLoad(ptrId, spv::NoPrecision);
builder.createStore(valueId, ptrId);
builder.createStore(valueId, ptrId, spv::MemoryAccessMask::Aligned | spv::MemoryAccessMask::MakePointerAvailableKHR,
                    spv::Scope::Device, 16); // store emits the scope word only for MakePointerAvailableKHR (load: MakePointerVisibleKHR)

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

spv::Id vec4 = builder.createConstructor(spv::NoPrecision, {scalarId}, vec4Ty);
spv::Id mat  = builder.createMatrixConstructor(spv::NoPrecision, srcs, mat4x4Ty);

// Swizzle
spv::Id swz = builder.createRvalueSwizzle(spv::NoPrecision, vec4Ty, vec, {2, 1, 0, 3});
spv::Id lswz= builder.createLvalueSwizzle(vec4Ty, target, source, {2, 1, 0, 3});

// Scalar promotion (in-place)
builder.promoteScalar(spv::NoPrecision, left, right);
spv::Id smeared = builder.smearScalar(spv::NoPrecision, scalarId, vec4Ty);
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
builder.accessChainStore(valueId, spv::Decoration::NonUniform,
                         spv::MemoryAccessMask::MaskNone, spv::Scope::Max, 0);
spv::Id lval = builder.accessChainGetLValue();
spv::Id inferred = builder.accessChainGetInferredType();
bool canBeLvalue = builder.isSpvLvalue();  // false for multi-component swizzles like .yx

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
builder.addExecutionMode(entry, spv::ExecutionMode::LocalSize, 64, 1, 1); // WorkgroupSize is a BuiltIn, not a Decoration
builder.addDecorationId(id, spv::Decoration::ArrayStrideIdEXT, strideId);
builder.addLinkageDecoration(id, "myFunc", spv::LinkageType::Export);
```

## Barriers

```cpp
builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device,
    spv::MemorySemanticsMask::UniformMemory | spv::MemorySemanticsMask::WorkgroupMemory);
builder.createMemoryBarrier(spv::Scope::Device, spv::MemorySemanticsMask::ImageMemory);
```

Group (subgroup) ops have no dedicated helper: add the capability, then emit the
`OpGroupNonUniform*` opcode through `createOp` (with `IdImmediate` literals as above) or
`createNoResultOp` — as done throughout `src/backends/common/spirv/spirv_codegen/instruction.cpp`
(e.g. `builder.addCapability(spv::Capability::GroupNonUniformVote);` then
`builder.createOp(spv::Op::OpGroupNonUniformAll, boolTy, {scope, val});`).

## Debug Info

### SPIR-V Standard (OpLine/OpSource)
```cpp
builder.setEmitSpirvDebugInfo();  // enables OpLine/OpSource tracking
builder.setDebugMainSourceFile("shader.glsl");
builder.setDebugSourceLocation(42, "shader.glsl");
builder.setSourceText(sourceText);
```

### NonSemantic Shader Debug Info
```cpp
builder.setEmitNonSemanticShaderDebugInfo(true);  // trackDebugInfo + NonSemantic import; emits DebugLine ext-insts (not OpLine)
spv::Id debugType = builder.getDebugType(spirvTypeId);
builder.enterLexicalBlock(line, column);
builder.leaveLexicalBlock();
builder.setupFunctionDebugInfo(func, "myFunc", paramTypes, paramNames);
spv::Id dbgGlobal = builder.createDebugGlobalVariable(debugType, "globalVar", varId);
spv::Id dbgLocal  = builder.createDebugLocalVariable(debugType, "localVar", argNumber);
spv::Id dbgDecl = builder.makeDebugDeclare(dbgLocal, ptrId);
spv::Id dbgVal = builder.makeDebugValue(dbgLocal, valueId);
```

Import string and enumerators are versioned but *not* named with a `100` suffix: the
OpExtInstImport name is built as `"NonSemantic.Shader.DebugInfo." + version`
(`SpvBuilder.cpp:2241`, `requireNonSemanticShaderDebugInfoVersion(unsigned)`), and the
opcode enumerators in `SPIRV/NonSemanticShaderDebugInfo.h` are
`NonSemanticShaderDebugInfoDebug*` (e.g. `NonSemanticShaderDebugInfoDebugLine`).

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

Both `postProcessCFG()` and `Function::dump()` traverse physical blocks with
`inReadableOrder()`, which assumes structured merge roles already nest. If an
outer selection merge is also an inner arm and then branches to the inner
merge, the physical graph exits the inner construct and re-enters it. The
traversal can initially mask that invalid topology by classifying the inner
merge as dead, replacing live code with `OpUnreachable`, and serializing it
before its dominator. Fix the producer's physical control-flow plan: preserve
the payload blocks but rotate the adjacent merge declarations so the inner
merge physically precedes the outer merge. Do not patch serialization order or
disable post-processing/validation around an invalid graph.

`OpSwitch` case literals are sized by the selector's `OpTypeInt`, not by the
generated operand-table class alone. A selector up to 32 bits uses one literal
word; a 64-bit selector uses two low-word-first literal words followed by one
target label ID. Disassemblers and binary walkers must resolve the selector
type and consume `ceil(bit_width / 32)` words per case before reading the label.
Never infer case boundaries by alternating one literal word and one ID.

Treat disassembly input as untrusted. Validate each instruction-local word
count before reading operands: reject zero, undersized, or module-truncated
instructions. When resolving an `OpSwitch` selector, also validate the mapped
defining instruction bounds and result ID; accept `OpTypeInt` only with its
exact four-word layout and a width of 8, 16, 32, or 64. Validate a directly
visited `OpTypeInt` before reading its width operand. The disassembler's fatal
path exits the process, so malformed-input regressions must run it in a child
process and assert the deterministic nonzero exit.

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
// the Function ctor already registers itself with the module (spvIR.h: `parent.addFunction(this)`)
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
// Treat spec constants, r-value parameters, and non-pointer/untyped values as r-values.
if (qualifier.isSpecConstant() || rValueParameters.find(symbol->getId()) != rValueParameters.end() ||
    (!builder.isPointerType(builder.getTypeId(id)) && !builder.isUntypedPointer(id)))
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
// operand is loaded r-value style; the access chain still points at the l-value
spv::Id operand = accessChainLoad(node->getOperand()->getType());
spv::Id one = builder.makeIntConstant(1);   // makeFloatConstant(1.0F) / makeInt8Constant(1) by basic type
// The traverser goes through its own createBinaryOperation() helper (glslang::TOperator
// EOpAdd/EOpSub + decorations), which in turn calls builder.createBinOp.
spv::Id result = createBinaryOperation(op, decorations, convertGlslangToSpvType(node->getType()),
                                       operand, one, node->getType().getBasicType());
builder.accessChainStore(result, TranslateNonUniformDecoration(builder.getAccessChain().coherentFlags));
builder.clearAccessChain();
if (node->getOp() == glslang::EOpPreIncrement || node->getOp() == glslang::EOpPreDecrement)
    builder.setAccessChainRValue(result);   // pre: yield the new value
else
    builder.setAccessChainRValue(operand);  // post: yield the old value
```

### visitUnary (Builtin / NoResult / ArrayLength)
```cpp
// Builtin
spv::Id result = builder.createBuiltinCall(resultType(), stdBuiltins, opcode, {operand});
// No-result (visitUnary; discard / terminate-invocation / demote are statement
// terminators emitted in visitBranch instead)
builder.createNoResultOp(spv::Op::OpAssumeTrueKHR, operand);
builder.createNoResultOp(spv::Op::OpEmitStreamVertex, operand);
builder.createNoResultOp(spv::Op::OpRayQueryTerminateKHR, operand);
// Array length
spv::Id len = builder.createArrayLength(builder.accessChainGetLValue(), member, bits);
len = builder.createUnaryOp(spv::Op::OpBitcast, builder.makeIntType(bits), len);
// Cooperative matrix/vector
spv::Id lenKHR = builder.createCooperativeMatrixLengthKHR(typeId);
spv::Id lenNV  = builder.createCooperativeMatrixLengthNV(typeId);
spv::Id lenVec = builder.getCooperativeVectorNumComponents(typeId);
// Tensor
spv::Id layout = builder.createOp(spv::Op::OpCreateTensorLayoutNV, resultType(), std::vector<spv::Id>{});
spv::Id view = builder.createOp(spv::Op::OpCreateTensorViewNV, resultType(), std::vector<spv::Id>{});
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
spv::Id result = builder.createTriOp(spv::Op::OpSelect, resultType, cond, trueVal, falseVal);
// Vector selection: for SPIR-V < 1.4 smear the scalar condition to the vector width;
// for SPIR-V >= 1.4 OpSelect accepts a scalar condition directly.
if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_4 && builder.isVector(trueVal)) {
    cond = builder.smearScalar(spv::NoPrecision, cond,
                               builder.makeVectorType(builder.makeBoolType(),
                                                      builder.getNumComponents(trueVal)));
}
// If aggregate decorations cause type mismatches, normalize with OpCopyLogical.
if (builder.getTypeId(trueVal) != resultType)
    trueVal = builder.createUnaryOp(spv::Op::OpCopyLogical, resultType, trueVal);
if (builder.getTypeId(falseVal) != resultType)
    falseVal = builder.createUnaryOp(spv::Op::OpCopyLogical, resultType, falseVal);
spv::Id result = builder.createTriOp(spv::Op::OpSelect, resultType, cond, trueVal, falseVal);
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
builder.setBuildPoint(&loop.continue_target); /* increment */ builder.createBranch(true, &loop.head);
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
builder.makeStatementTerminator(spv::Op::OpTerminateInvocation, "post-terminate-invocation");
builder.createNoResultOp(spv::Op::OpDemoteToHelperInvocationEXT);
builder.makeStatementTerminator(spv::Op::OpTerminateRayKHR, "post-terminateRayKHR");
builder.makeStatementTerminator(spv::Op::OpIgnoreIntersectionKHR, "post-ignoreIntersectionKHR");
```

### visitConstantUnion
```cpp
int nextConst = 0;
spv::Id constant = createSpvConstantFromConstUnionArray(node->getType(), node->getConstArray(), nextConst, false);
builder.clearAccessChain();
builder.setAccessChainRValue(constant);
```

### visitVariableDecl
```cpp
builder.setDebugSourceLocation(node->getDeclSymbol()->getLoc().line,
                               node->getDeclSymbol()->getLoc().getFilename());
```

### visitFunctions
```cpp
// No direct builder usage; drives traversal of the translation unit.
```
