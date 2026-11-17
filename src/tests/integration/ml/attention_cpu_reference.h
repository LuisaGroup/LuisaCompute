// Multi-Head Latent Attention (MLA) Example -- CPU reference implementation.

#pragma once

#include <luisa/core/stl.h>

#include "attention_host_data.h"

namespace mla {

// Run the multi-threaded CPU reference for the selected attention path and
// return the output tensor O of size qkv_size.
[[nodiscard]] luisa::vector<float> run_cpu_reference(const AttentionHostData &data, bool use_mla);

}// namespace mla
