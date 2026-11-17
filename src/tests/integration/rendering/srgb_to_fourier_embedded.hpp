#pragma once
#ifdef LUISA_BIN_2_OBJ
#include <cstdint>
extern "C" {
extern const uint8_t _binary_SRGBToFourierEvenPacked_dat_start[];
extern const uint8_t _binary_SRGBToFourierEvenPacked_dat_end[];
}
#define luisa_compute_SRGBToFourierEvenPacked ((const unsigned char *)_binary_SRGBToFourierEvenPacked_dat_start)
#define luisa_compute_SRGBToFourierEvenPacked_size ((unsigned long long)(_binary_SRGBToFourierEvenPacked_dat_end - _binary_SRGBToFourierEvenPacked_dat_start))
#else
#include "srgb_to_fourier_embedded.h"
#endif
