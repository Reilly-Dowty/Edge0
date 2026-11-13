// Latent MoE: the one session-level D/U device pair, stored by the engine's LatentTab and pulled by the
// prefill and verify arms inside their own init (no per-object external wiring to miss).  A non-latent pack
// never stores a pair, so both accessors stay null.
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace strata {
namespace core {

// edge0-multigpu: one D/U pair PER DEVICE.  A layer split runs each stage on its own card, and a card without P2P
// cannot read another card's VRAM, so every accessor returns the slot of the CURRENT device (the stage's OnDevice).
// A stage's pointer is its own upload offset by -lb * bytes: callers keep indexing with the global layer number.
inline int latent_share_dev() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess) { (void) cudaGetLastError(); d = 0; }
    return (d < 0 || d >= 16) ? 0 : d;
}
inline uint8_t*& latent_shared_d() { static uint8_t* p[16] = {}; return p[latent_share_dev()]; }
inline uint8_t*& latent_shared_u() { static uint8_t* p[16] = {}; return p[latent_share_dev()]; }

inline const uint8_t*& latent_xl_current() { static const uint8_t* p = nullptr; return p; }
inline const uint32_t*& latent_xlseq_current() { static const uint32_t* p = nullptr; return p; }
inline const uint8_t*& latent_act0_current() { static const uint8_t* p = nullptr; return p; }

}  // namespace core
}  // namespace strata
