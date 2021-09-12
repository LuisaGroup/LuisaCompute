//
// Created by Mike Smith on 2021/9/13.
//

#include <core/dynamic_module.h>
#include <serialize/interface.h>

using namespace luisa;
using namespace luisa::serialize;

int main(int argc, char *argv[]) {
    auto bin_dir = std::filesystem::canonical(argv[0]).parent_path();
    DynamicModule module{bin_dir, "luisa-compute-serialize-json"};
    auto factory = std::invoke(module.function<DatabaseFactory>(database_factory_symbol));
    auto database = factory->CreateDatabase();
    auto dict = database->CreateDict();
    auto array = database->CreateArray();
    array->Add("Hello");
    array->Add(0);
    dict->Set("Hello", array.get());
    LUISA_INFO("{}", database->Print());
}
