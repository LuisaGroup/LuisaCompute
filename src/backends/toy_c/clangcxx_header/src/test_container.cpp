#include <luisa/std.hpp>
#include <container/vector.hpp>
#include <container/string.hpp>
#include <luisa/printer.hpp>
using namespace luisa::shader;

[[kernel_1d(1)]] int kernel(){
    // vector<int> vec;
    // vec.emplace_back(1);
    // vec.emplace_back(2);
    // vec.emplace_back(666);
    // for(int i = 0; i < vec.size(); ++i){
    //     device_log("{}", vec[i]);
    // }
    // vec.pop_back();
    // for(int i = 0; i < vec.size(); ++i){
    //     device_log("{}", vec[i]);
    // }
    // // Free allocated vector ptr
    // dispose(vec);

    // vector<string> str_vec;
    // str_vec.emplace_back(to_strview("hello"));
    // str_vec.emplace_back(to_strview("world"));
    // // Will free string's memory too
    // dispose(str_vec);
    return 0;
}