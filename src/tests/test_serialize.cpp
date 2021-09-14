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
    auto factory = module.invoke<DatabaseFactory>(database_factory_symbol);
    auto database = factory->CreateDatabase();
    auto root = database->GetRootNode();
    auto dict = database->CreateDict();
    auto array = database->CreateArray();
    array->Add("World");
    array->Add(0);
    dict->Set("Hello", std::move(array));
    root->Set("good", std::move(dict));
    auto s = database->Print();
    LUISA_INFO("{}", s);
}
