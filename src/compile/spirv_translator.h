//
// Created by Mike Smith on 2022/6/29.
//

#pragma once

#include <ast/function_builder.h>

namespace luisa::compute {

/// @brief Translate LuisaCompute ASTs to SPIR-V.
class SPIRVTranslator : StmtVisitor {
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const CommentStmt *stmt) override;
};

}// namespace luisa::compute
