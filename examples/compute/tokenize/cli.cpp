#include "cli_functions.h"
#include "cli_entry.h"
#include <iostream>
#include <yyjson.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/logging.h>

using FunctionListType = luisa::unordered_map<
    luisa::string,
    luisa::move_only_function<luisa::string(cli::ArgumentList)>>;

int run_cli() {
    luisa::log_level_error();
    std::cout << "Input:" << std::endl;
    luisa::fiber::scheduler global_scheduler;
    FunctionListType functions;
    functions.emplace("add_builder", cli::cmd_add_builder);
    functions.emplace("remove_builder", cli::cmd_remove_builder);
    functions.emplace("search", cli::cmd_search);

    luisa::string line;
    while (std::getline(std::cin, line)) {
        if (line == "exit") { break; }

        yyjson_read_err err{};
        yyjson_doc *doc = yyjson_read_opts(
            const_cast<char *>(line.data()), line.size(),
            YYJSON_READ_NOFLAG, nullptr, &err);
        if (!doc) {
            LUISA_WARNING("JSON parse error at pos {}: {}", err.pos, err.msg);
            continue;
        }

        auto root = yyjson_doc_get_root(doc);
        auto func_val = yyjson_obj_get(root, "func");
        auto args_val = yyjson_obj_get(root, "args");

        if (!yyjson_is_str(func_val)) {
            LUISA_WARNING("Missing or invalid 'func' field");
            yyjson_doc_free(doc);
            continue;
        }

        cli::ArgumentList args;
        if (args_val && yyjson_is_arr(args_val)) {
            size_t idx = 0, max = 0;
            yyjson_val *v = nullptr;
            yyjson_arr_foreach(args_val, idx, max, v) {
                if (yyjson_is_str(v)) {
                    args.emplace_back(yyjson_get_str(v));
                } else {
                    char *s = yyjson_val_write(v, YYJSON_WRITE_NOFLAG, nullptr);
                    if (s) {
                        args.emplace_back(s);
                        free(s);
                    }
                }
            }
        }

        const char *func_name = yyjson_get_str(func_val);
        auto it = functions.find(func_name);
        if (it == functions.end()) {
            LUISA_WARNING("Function '{}' not registered", func_name);
            yyjson_doc_free(doc);
            continue;
        }

        auto result = it->second(std::move(args));
        std::cout << result << "==e6b7e03aa02b4ffe==" << std::endl;

        yyjson_doc_free(doc);
    }

    return 0;
}
