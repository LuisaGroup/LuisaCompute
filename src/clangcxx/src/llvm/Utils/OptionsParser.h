//===- CommonOptionsParser.h - common options for clang tools -*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CommonOptionsParser class used to parse common
//  command-line options for clang tools, so that they can be run as separate
//  command-line applications with a consistent common interface for handling
//  compilation database and input files.
//
//  It provides a common subset of command-line options, common algorithm
//  for locating a compilation database and source files, and help messages
//  for the basic command-line interface.
//
//  It creates a CompilationDatabase and reads common command-line options.
//
//  This class uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

namespace luisa::clangcxx
{
using namespace clang;
using namespace clang::tooling;

class OptionsParser
{
protected:
    /// Parses command-line, initializes a compilation database.
    ///
    /// This constructor can change argc and argv contents, e.g. consume
    /// command-line options used for creating FixedCompilationDatabase.
    ///
    /// All options not belonging to \p Category become hidden.
    ///
    /// It also allows calls to set the required number of positional parameters.
    OptionsParser(int& argc, const char** argv, llvm::cl::NumOccurrencesFlag OccurrencesFlag, llvm::cl::OptionCategory& Category, const char* Overview = nullptr);

public:
    /// A factory method that is similar to the above constructor, except
    /// this returns an error instead exiting the program on error.
    static llvm::Expected<OptionsParser> create(int& argc, const char** argv, llvm::cl::NumOccurrencesFlag OccurrencesFlag, llvm::cl::OptionCategory& Category, const char* Overview = nullptr);

    /// Returns a reference to the loaded compilations database.
    CompilationDatabase& getCompilations()
    {
        return *Compilations;
    }

    /// Returns a list of source file paths to process.
    [[nodiscard]] const std::vector<std::string>& getSourcePathList() const
    {
        return SourcePathList;
    }

    /// Returns the argument adjuster calculated from "--extra-arg" and
    //"--extra-arg-before" options.
    ArgumentsAdjuster getArgumentsAdjuster()
    {
        return Adjuster;
    }

    static const char* const HelpMessage;

private:
    OptionsParser() = default;

    llvm::Error init(int& argc, const char** argv, llvm::cl::NumOccurrencesFlag OccurrencesFlag, llvm::cl::OptionCategory& Category, const char* Overview);

    std::unique_ptr<CompilationDatabase> Compilations;
    std::vector<std::string> SourcePathList;
    ArgumentsAdjuster Adjuster;
};

} // namespace luisa::clangcxx
