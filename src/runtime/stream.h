//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

namespace luisa::compute {

class Stream {

private:
    virtual void _dispatch(const class BufferCopyCommand &) = 0;
    virtual void _dispatch(const class BufferUploadCommand &) = 0;
    virtual void _dispatch(const class BufferDownloadCommand &) = 0;

public:
    template<typename Cmd>
    Stream &operator<<(Cmd &&cmd) {
        _dispatch(std::forward<Cmd>(cmd));
        return *this;
    }
    
};

}// namespace luisa::compute
