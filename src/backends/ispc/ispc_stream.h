//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <core/thread_pool.h>
#include <runtime/command_list.h>

namespace luisa::compute::ispc {

class ISPCEvent;

class ISPCStream {

private:
    ThreadPool _pool;

public:
    ISPCStream() noexcept = default;
    void synchronize() noexcept { _pool.synchronize(); }
    void dispatch(const CommandList &cmd_list) noexcept;
    void signal(ISPCEvent *event) noexcept;
    void wait(ISPCEvent *event) noexcept;
};

}// namespace luisa::compute::ispc
