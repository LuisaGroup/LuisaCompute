// Variable Naming

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"

namespace lc::hlsl {

// Get variable name by components
void CodegenUtility::GetVariableName(Function f, Variable::Tag type, uint id, vstd::StringBuilder &str) {
    switch (type) {
        case Variable::Tag::BLOCK_ID:
            LUISA_ASSERT(!opt->isRaster, "block id only allowed in compute shader");
            str << "grpId"sv;
            break;
        case Variable::Tag::DISPATCH_ID:
            LUISA_ASSERT(!opt->isRaster, "dispatch id only allowed in compute shader");
            str << "dspId"sv;
            break;
        case Variable::Tag::THREAD_ID:
            LUISA_ASSERT(!opt->isRaster, "thread id only allowed in compute shader");
            str << "thdId"sv;
            break;
        case Variable::Tag::DISPATCH_SIZE:
            LUISA_ASSERT(!opt->isRaster, "dispatch_size only allowed in compute shader");
            if (opt->funcType == CodegenStackData::FuncType::Kernel) {
                if (opt->isSpirv)
                    str << "dsp_c.v.xyz"sv;
                else
                    str << "dsp_c.xyz"sv;
            } else {
                str << "dsp_c"sv;
            }
            break;
        case Variable::Tag::KERNEL_ID:
            if (opt->isRaster) {
                str << "primId"sv;
            } else {
                if (opt->funcType != CodegenStackData::FuncType::Callable) {
                    if (opt->isSpirv)
                        str << "dsp_c.v.w"sv;
                    else
                        str << "dsp_c.w"sv;
                } else {
                    str << "ker"sv;
                }
            }
            break;
        case Variable::Tag::RASTER_OBJECT_ID:
            LUISA_ASSERT(opt->isRaster, "object id only allowed in raster shader");
            if (opt->funcType != CodegenStackData::FuncType::Callable) {
                if (opt->isSpirv)
                    str << "obj_id.v"sv;
                else
                    str << "obj_id"sv;
            } else {
                str << "ker"sv;
            }

            break;
        case Variable::Tag::RASTER_BARYCENTRICS:
            LUISA_ASSERT(opt->isRaster, "barycentrics only allowed in raster shader");
            str << "bary"sv;
            opt->pixelUseBarycentric = true;
            break;
        case Variable::Tag::WARP_LANE_COUNT:
            LUISA_ASSERT(!opt->isRaster, "warp ops only allowed in compute shader");
            if (opt->funcType == CodegenStackData::FuncType::Callable) {
                str << "_wrpct"sv;
            } else {
                str << "WaveGetLaneCount()"sv;
            }
            break;
        case Variable::Tag::WARP_LANE_ID:
            LUISA_ASSERT(!opt->isRaster, "warp ops only allowed in compute shader");
            if (opt->funcType == CodegenStackData::FuncType::Callable) {
                str << "_wrpid"sv;
            } else {
                str << "WaveGetLaneIndex()"sv;
            }
            break;
        case Variable::Tag::LOCAL:
            switch (opt->funcType) {
                case CodegenStackData::FuncType::Kernel:
                case CodegenStackData::FuncType::Vert:
                    if (id == opt->appdataId) {
                        str << "vv"sv;
                    } else {

                        if (opt->arguments.find(id) != opt->arguments.end()) {
                            id += opt->argOffset;
                            str << "a.l"sv;
                        } else {
                            auto custom_name = f.get_variable_name(id);
                            if (!custom_name.empty())
                                str << custom_name << '_';
                            str << 'l';
                        }
                        vstd::to_string(id, str);
                    }
                    break;
                case CodegenStackData::FuncType::Pixel: {
                    auto ite = opt->arguments.find(id);
                    if (ite == opt->arguments.end()) {
                        auto custom_name = f.get_variable_name(id);
                        if (!custom_name.empty())
                            str << custom_name << '_';
                        str << 'l';
                        vstd::to_string(id, str);
                    } else {
                        if (ite->second == 0) {
                            if (opt->pixelFirstArgIsStruct) {
                                str << 'p';
                            } else {
                                str << "p.v0"sv;
                            }
                        } else {
                            id += opt->argOffset;
                            str << "a.l"sv;
                            vstd::to_string(id, str);
                        }
                    }
                } break;
                default: {
                    auto custom_name = f.get_variable_name(id);
                    if (!custom_name.empty())
                        str << custom_name << '_';
                    str << 'l';
                    vstd::to_string(id, str);
                }
            }
            break;
        case Variable::Tag::SHARED: {
            auto custom_name = f.get_variable_name(id);
            str << custom_name;
            str << "_s";
            vstd::to_string(f.hash(), str);
            str << '_';
            vstd::to_string(id, str);
        } break;
        case Variable::Tag::REFERENCE: {
            auto custom_name = f.get_variable_name(id);
            if (!custom_name.empty())
                str << custom_name << '_';
            str << 'r';
            vstd::to_string(id, str);
        } break;
        case Variable::Tag::BUFFER: {
            auto custom_name = f.get_variable_name(id);
            str << custom_name;
            str << "_b"sv;
            vstd::to_string(id, str);
        } break;
        case Variable::Tag::TEXTURE: {
            auto custom_name = f.get_variable_name(id);
            str << custom_name;
            str << "_t"sv;
            vstd::to_string(id, str);
        } break;
        case Variable::Tag::BINDLESS_ARRAY: {
            auto custom_name = f.get_variable_name(id);
            str << custom_name;
            str << "_ba"sv;
            vstd::to_string(id, str);
        } break;
        case Variable::Tag::ACCEL: {
            auto custom_name = f.get_variable_name(id);
            str << custom_name;
            str << "_ac"sv;
            vstd::to_string(id, str);
        } break;
        default: {
            auto custom_name = f.get_variable_name(id);
            if (!custom_name.empty())
                str << custom_name << '_';
            str << 'v';
            vstd::to_string(id, str);
        } break;
    }
}

// Get variable name from variable object
void CodegenUtility::GetVariableName(Function f, Variable const &type, vstd::StringBuilder &str) {
    GetVariableName(f, type.tag(), type.uid(), str);
}

}// namespace lc::hlsl
