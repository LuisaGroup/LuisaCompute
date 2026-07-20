#pragma once

#include <span>
#include <memory>

#include <luisa/core/basic_types.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/usage.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}// namespace luisa::compute

namespace luisa::compute::hip {

class HIPCommandEncoder;
class HIPShaderPrinter;

class HIPShader {

private:
    luisa::unique_ptr<HIPShaderPrinter> _printer;
    luisa::vector<Usage> _argument_usages;
    luisa::string _name;
    mutable spin_mutex _name_mutex;

private:
    virtual void _launch(HIPCommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept = 0;

public:
    HIPShader(luisa::unique_ptr<HIPShaderPrinter> printer,
              luisa::vector<Usage> arg_usages) noexcept;
    virtual ~HIPShader() noexcept;
    HIPShader(HIPShader &&) noexcept = delete;
    HIPShader(const HIPShader &) noexcept = delete;
    HIPShader &operator=(HIPShader &&) noexcept = delete;
    HIPShader &operator=(const HIPShader &) noexcept = delete;
    [[nodiscard]] Usage argument_usage(size_t i) const noexcept;
    [[nodiscard]] auto printer() const noexcept { return _printer.get(); }
    [[nodiscard]] virtual void *handle() const noexcept = 0;
    void launch(HIPCommandEncoder &encoder,
                ShaderDispatchCommand *command) const noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::hip
