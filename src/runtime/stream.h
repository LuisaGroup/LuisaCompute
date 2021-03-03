//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>
#include <runtime/command.h>

namespace luisa::compute {

class Stream {

private:
    virtual void _dispatch(const BufferCopyCommand &) = 0;
    virtual void _dispatch(const BufferUploadCommand &) = 0;
    virtual void _dispatch(const BufferDownloadCommand &) = 0;
    virtual void _dispatch(const KernelLaunchCommand &) = 0;

public:
    template<typename Cmd>
    Stream &operator<<(Cmd &&cmd) {
        _dispatch(std::forward<Cmd>(cmd));
        return *this;
    }
    
};

}// namespace luisa::compute
