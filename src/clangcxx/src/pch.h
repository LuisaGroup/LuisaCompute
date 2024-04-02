#pragma once

#ifndef LUISA_COMPUTE_CLANGCXX_PCH_H
#define LUISA_COMPUTE_CLANGCXX_PCH_H

#include <luisa/vstl/common.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/logging.h>
#include <luisa/core/basic_traits.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/concepts.h>
#include <luisa/core/magic_enum.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/magic_enum.h>

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>

#include <luisa/ast/op.h>
#include <luisa/ast/callable_library.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/rtx/ray_query.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Path.h>

#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/RecordLayout.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecursiveASTVisitor.h>

#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

#include <utility>
#include <iostream>

#endif
