#ifndef __FC_MMFILE_H__
#define __FC_MMFILE_H__

#include <cstdint>
#include <filesystem>
#include <stdexcept>

#if _WIN32 || _WIN64
#include "fc/mmfile_win.h"
#else
#include "fc/mmfile_nix.h"
#endif

namespace frozenca {

class MemoryMappedFile {
 public:
  static inline constexpr std::size_t new_file_size_ =
      MemoryMappedFileImpl::new_file_size_;
  using handle_type = MemoryMappedFileImpl::handle_type;
  using path_type = MemoryMappedFileImpl::path_type;

 private:
  MemoryMappedFileImpl impl_;

 public:
  MemoryMappedFile(const std::filesystem::path &path,
                   std::size_t init_file_size = new_file_size_,
                   bool trunc = false)
      : impl_{path, init_file_size, trunc} {}

  ~MemoryMappedFile() noexcept = default;

 public:
  void resize(std::size_t new_size) { impl_.resize(new_size); }

  [[nodiscard]] std::size_t size() const noexcept { return impl_.size(); }

  [[nodiscard]] void *data() noexcept { return impl_.data(); }

  [[nodiscard]] const void *data() const noexcept { return impl_.data(); }

  friend bool operator==(const MemoryMappedFile &mmfile1,
                         const MemoryMappedFile &mmfile2) {
    return mmfile1.impl_ == mmfile2.impl_;
  }

  friend bool operator!=(const MemoryMappedFile &mmfile1,
                         const MemoryMappedFile &mmfile2) {
    return !(mmfile1 == mmfile2);
  }
};

}  // namespace frozenca

#endif  //__FC_MMFILE_H__
