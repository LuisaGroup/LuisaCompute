//===--- CommonOptionsParser.cpp - common options for clang tools ---------===//
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

#include "OptionsParser.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#include <filesystem>
#include <iostream>

using namespace clang::tooling;
using namespace llvm;
using namespace luisa::clangcxx;

const char *const OptionsParser::HelpMessage =
    "\n"
    "<source0> ... specify the paths of source files. These paths are\n"
    "\tlooked up in the compile command database. If the path of a file is\n"
    "\tabsolute, it needs to point into CMake's source tree. If the path is\n"
    "\trelative, the current working directory needs to be in the CMake\n"
    "\tsource tree and the file must be in a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in the relative files will be\n"
    "\tautomatically removed, but the rest of a relative path must be a\n"
    "\tsuffix of a path in the compile command database.\n"
    "\n";
// static void replaceAll(std::string &str, const std::string &from,
//                        const std::string &to) {
//     if (from.empty())
//         return;
//     size_t start_pos = 0;
//     while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
//         str.replace(start_pos, from.length(), to);
//         start_pos += to.length();// In case 'to' contains 'from', like replacing
//                                  // 'x' with 'yx'
//     }
// }

llvm::Error OptionsParser::init(int &argc, const char **argv,
                                llvm::cl::NumOccurrencesFlag OccurrencesFlag,
                                cl::OptionCategory &Category,
                                const char *Overview) {
    static cl::list<std::string> SourcePaths(
        cl::Positional, cl::desc("<source0> [... <sourceN>]"), OccurrencesFlag,
        cl::cat(Category), cl::sub(*cl::AllSubCommands));

    static cl::list<std::string> ArgsAfter(
        "extra-arg",
        cl::desc("Additional argument to append to the compiler command line"),
        cl::cat(Category), cl::sub(*cl::AllSubCommands));

    static cl::list<std::string> ArgsBefore(
        "extra-arg-before",
        cl::desc("Additional argument to prepend to the compiler command line"),
        cl::cat(Category), cl::sub(*cl::AllSubCommands));

    cl::ResetAllOptionOccurrences();
    cl::HideUnrelatedOptions(Category);

    std::string ErrorMessage;
    const char *const *DoubleDash = std::find(argv, argv + argc, StringRef("--"));
    if (DoubleDash != argv + argc) {
        if (DoubleDash[1][0] == '@') {
            llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
                llvm::MemoryBuffer::getFile(DoubleDash[1] + 1);
            if (std::error_code Result = File.getError()) {
                ErrorMessage = "Error while opening fixed database: " + Result.message();
                return llvm::make_error<llvm::StringError>(ErrorMessage,
                                                           llvm::inconvertibleErrorCode());
            }
            Compilations = FixedCompilationDatabase::loadFromBuffer(".", (*File)->getBuffer(), ErrorMessage);
            argc = DoubleDash - argv;
        } else {
            Compilations = FixedCompilationDatabase::loadFromCommandLine(argc, argv, ErrorMessage);
        }
    }
    if (!ErrorMessage.empty())
        ErrorMessage.append("\n");

    llvm::raw_string_ostream OS(ErrorMessage);
    // Stop initializing if command-line option parsing failed.
    if (!cl::ParseCommandLineOptions(argc, argv, Overview, &OS)) {
        OS.flush();
        return llvm::make_error<llvm::StringError>(ErrorMessage, llvm::inconvertibleErrorCode());
    }
    cl::PrintOptionValues();

    // Not loaded, try parse from commandline
    SourcePathList = SourcePaths;
    if (SourcePaths.empty()) {
        auto thisFile = std::filesystem::current_path();
        auto testFile = thisFile.parent_path().parent_path().parent_path().parent_path() / "tests" / "test.cpp";
        SourcePathList.emplace_back(testFile.string());
    }

    if (SourcePathList.empty()) {
        return llvm::Error::success();
    }

    // TODO: Create CompilationDatabase
    if (!Compilations) {
        Compilations = std::make_unique<FixedCompilationDatabase>(".", std::vector<std::string>());
        if (!Compilations) {
            llvm::errs() << "Error while trying to load a compilation database:\n"
                         << ErrorMessage << "Running without flags.\n";
        }
    }
    auto AdjustingCompilations = std::make_unique<ArgumentsAdjustingCompilations>(std::move(Compilations));
    Adjuster = getInsertArgumentAdjuster(ArgsBefore, ArgumentInsertPosition::BEGIN);
    Adjuster = combineAdjusters(
        std::move(Adjuster),
        getInsertArgumentAdjuster(ArgsAfter, ArgumentInsertPosition::END));
    AdjustingCompilations->appendArgumentsAdjuster(Adjuster);
    Compilations = std::move(AdjustingCompilations);
    return llvm::Error::success();
}

llvm::Expected<OptionsParser> OptionsParser::create(
    int &argc, const char **argv, llvm::cl::NumOccurrencesFlag OccurrencesFlag,
    llvm::cl::OptionCategory &Category, const char *Overview) {
    OptionsParser Parser;
    llvm::Error Err = Parser.init(argc, argv, OccurrencesFlag, Category, Overview);
    if (Err)
        return std::move(Err);
    return std::move(Parser);
}

OptionsParser::OptionsParser(int &argc, const char **argv,
                             llvm::cl::NumOccurrencesFlag OccurrencesFlag,
                             cl::OptionCategory &Category,
                             const char *Overview) {
    llvm::Error Err = init(argc, argv, OccurrencesFlag, Category, Overview);
    if (Err) {
        llvm::report_fatal_error(
            Twine("CommonOptionsParser: failed to parse command-line arguments. ") +
            llvm::toString(std::move(Err)));
    }
}
