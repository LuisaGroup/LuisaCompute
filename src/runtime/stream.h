//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

namespace luisa::compute {

class Stream {

private:
#define LUISA_MAKE_STREAM_DISPATCH(CommandType) \
    friend class CommandType;                   \
    virtual void _dispatch(const class CommandType &) = 0;
    LUISA_MAKE_STREAM_DISPATCH(BufferCopyCommand)
    LUISA_MAKE_STREAM_DISPATCH(BufferUploadCommand)
    LUISA_MAKE_STREAM_DISPATCH(BufferDownloadCommand)
#undef LUISA_MAKE_STREAM_DISPATCH

public:
    template<typename Cmd, typename = decltype(std::declval<Cmd>().commit(std::declval<Stream>()))>
    Stream &operator<<(Cmd &&command) {
        command.commit(*this);
        return *this;
    }
    
    
};

}// namespace luisa::compute
