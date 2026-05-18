#ifndef SAMPLER_LLAMA_BRIDGE_H
#define SAMPLER_LLAMA_BRIDGE_H

#include "random_token.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t rp_sample_from_logits_llama_cpp(rp_sampler_state *state,
                                         const rp_sampling_config *config,
                                         const uint32_t *token_ids,
                                         const float *logits,
                                         size_t vocab_size);

#ifdef __cplusplus
}
#endif

#endif
