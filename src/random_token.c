#include "random_token.h"

#include "sampler_llama_bridge.h"

#include <string.h>

enum {
  RP_GLOBAL_SEED = 0x12345678u,
  RP_RNG_MASK = 0x7fffffffu,
  RP_DEFAULT_TOP_K = 5,
};

static const uint32_t k_offline_token_table[] = {
    128000u, 128001u, 128002u, 128003u, 128004u, 128005u, 128006u, 128007u,
    128008u, 128009u, 128010u, 128011u, 128012u, 128013u, 128014u, 128015u};

static const uint32_t k_offline_logits_token_ids[] = {128000u, 128001u, 128002u,
                                                       128003u, 128004u, 128005u,
                                                       128006u, 128007u};

static const float k_offline_logits[][8] = {
    {2.1f, 1.6f, 0.5f, -0.4f, -1.0f, -2.0f, 0.9f, 1.3f},
    {1.7f, 1.3f, 0.4f, -0.2f, -1.4f, -1.9f, 1.2f, 0.8f},
    {1.9f, 1.5f, 0.6f, -0.3f, -1.2f, -2.1f, 0.7f, 1.1f},
    {2.0f, 1.4f, 0.2f, -0.6f, -1.5f, -2.0f, 0.8f, 1.0f},
    {1.8f, 1.1f, 0.3f, -0.1f, -1.1f, -2.3f, 1.4f, 0.9f},
    {2.2f, 1.2f, 0.7f, -0.5f, -1.0f, -1.8f, 0.6f, 1.5f},
};

static uint32_t g_state = RP_GLOBAL_SEED;

static uint32_t rp_next_state(uint32_t current) {
  return (1103515245u * current + 12345u) & RP_RNG_MASK;
}

void rp_reset(void) { g_state = RP_GLOBAL_SEED; }

uint32_t rp_next_token_id(void) {
  const size_t count = rp_token_count();
  g_state = rp_next_state(g_state);
  return k_offline_token_table[g_state % count];
}

size_t rp_token_count(void) {
  return sizeof(k_offline_token_table) / sizeof(k_offline_token_table[0]);
}

const uint32_t *rp_token_table(void) { return k_offline_token_table; }

void rp_default_sampling_config(rp_sampling_config *config) {
  if (config == NULL) {
    return;
  }
  config->seed = RP_GLOBAL_SEED;
  config->temperature = 0.8f;
  config->top_k = RP_DEFAULT_TOP_K;
  config->top_p = 0.9f;
  config->min_p = 0.05f;
  config->repeat_penalty = 1.1f;
  config->frequency_penalty = 0.1f;
  config->presence_penalty = 0.05f;
  config->history_window = 64u;
}

void rp_sampler_init(rp_sampler_state *state, const rp_sampling_config *config) {
  if (state == NULL || config == NULL) {
    return;
  }
  state->rng_state = config->seed;
  state->history_len = 0;
  memset(state->history, 0, sizeof(state->history));
}

uint32_t rp_sample_from_logits(rp_sampler_state *state,
                               const rp_sampling_config *config,
                               const uint32_t *token_ids, const float *logits,
                               size_t vocab_size) {
  return rp_sample_from_logits_llama_cpp(state, config, token_ids, logits, vocab_size);
}

size_t rp_offline_logits_vocab_size(void) {
  return sizeof(k_offline_logits_token_ids) / sizeof(k_offline_logits_token_ids[0]);
}

size_t rp_offline_logits_step_count(void) {
  return sizeof(k_offline_logits) / sizeof(k_offline_logits[0]);
}

const uint32_t *rp_offline_logits_token_ids(void) {
  return k_offline_logits_token_ids;
}

const float *rp_offline_logits_for_step(size_t step) {
  const size_t steps = rp_offline_logits_step_count();
  if (steps == 0) {
    return NULL;
  }
  return k_offline_logits[step % steps];
}
