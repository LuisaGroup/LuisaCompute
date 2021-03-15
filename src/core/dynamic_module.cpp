#include <core/dynamic_module.h>

namespace luisa {

DynamicModule &DynamicModule::operator=(DynamicModule &&rhs) noexcept {
    _handle = rhs._handle;
    rhs._handle = nullptr;
    return *this;
}

DynamicModule::DynamicModule(DynamicModule &&another) noexcept
    : _handle{another._handle} { another._handle = nullptr; }

DynamicModule::~DynamicModule() noexcept { dynamic_module_destroy(_handle); }

DynamicModule::DynamicModule(const std::filesystem::path &path) noexcept
    : _handle{dynamic_module_load(path)} {}

}// namespace luisa
