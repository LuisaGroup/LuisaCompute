#pragma once


#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

    class Module;
    class Function;


    struct Canonicalize_Control_Flow_Info{
        size_t lowered_loop_count{0u};
        size_t skipped_loop_count{0u};
    };


    [[nodiscard]] LUISA_XIR_API Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Module(Module* module);
    [[nodiscard]] LUISA_XIR_API Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Function(Function* func);


    
}
