#ifndef RANDOM_TOKEN_H
#define RANDOM_TOKEN_H

#include <stddef.h>
#include <stdint.h>

#define RP_MAX_HISTORY 256u

void rp_reset(void);
uint32_t rp_next_token_id(void);
size_t rp_token_count(void);
const uint32_t *rp_token_table(void);

typedef struct {
  uint32_t seed;
  float temperature;
  int top_k;
  float top_p;
  float min_p;
  float repeat_penalty;
  float frequency_penalty;
  float presence_penalty;
  // 0 means use RP_MAX_HISTORY.
  size_t history_window;
} rp_sampling_config;

typedef struct {
  uint32_t rng_state;
  uint32_t history[RP_MAX_HISTORY];
  size_t history_len;
} rp_sampler_state;

void rp_default_sampling_config(rp_sampling_config *config);
void rp_sampler_init(rp_sampler_state *state, const rp_sampling_config *config);
uint32_t rp_sample_from_logits(rp_sampler_state *state,
                               const rp_sampling_config *config,
                               const uint32_t *token_ids, const float *logits,
                               size_t vocab_size);

size_t rp_offline_logits_vocab_size(void);
size_t rp_offline_logits_step_count(void);
const uint32_t *rp_offline_logits_token_ids(void);
const float *rp_offline_logits_for_step(size_t step);

#endif
