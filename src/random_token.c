#include "random_token.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
  RP_GLOBAL_SEED = 0x12345678u,
  RP_RNG_MASK = 0x7fffffffu,
  RP_DEFAULT_TOP_K = 5,
};

typedef struct {
  uint32_t token_id;
  double logit;
  double prob;
  int keep;
} rp_candidate;

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

static double rp_next_uniform(rp_sampler_state *state) {
  state->rng_state = rp_next_state(state->rng_state);
  return (double)state->rng_state / (double)(RP_RNG_MASK + 1u);
}

static size_t rp_count_token_in_history(const rp_sampler_state *state,
                                        uint32_t token_id,
                                        size_t history_window) {
  size_t count = 0;
  size_t start = 0;
  if (state->history_len > history_window) {
    start = state->history_len - history_window;
  }
  for (size_t i = start; i < state->history_len; ++i) {
    if (state->history[i] == token_id) {
      ++count;
    }
  }
  return count;
}

static void rp_sort_indices_desc_by_score(size_t *indices, const double *scores,
                                          size_t count) {
  for (size_t i = 1; i < count; ++i) {
    size_t key = indices[i];
    size_t j = i;
    while (j > 0 && scores[indices[j - 1]] < scores[key]) {
      indices[j] = indices[j - 1];
      --j;
    }
    indices[j] = key;
  }
}

static size_t rp_count_kept(const rp_candidate *candidates, size_t vocab_size) {
  size_t kept = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (candidates[i].keep) {
      ++kept;
    }
  }
  return kept;
}

static void rp_apply_top_k(rp_candidate *candidates, size_t vocab_size, int top_k) {
  if (top_k <= 0 || (size_t)top_k >= vocab_size) {
    return;
  }

  size_t *indices = (size_t *)malloc(vocab_size * sizeof(size_t));
  if (indices == NULL) {
    return;
  }
  for (size_t i = 0; i < vocab_size; ++i) {
    indices[i] = i;
  }

  double *scores = (double *)malloc(vocab_size * sizeof(double));
  if (scores == NULL) {
    free(indices);
    return;
  }
  for (size_t i = 0; i < vocab_size; ++i) {
    scores[i] = candidates[i].logit;
  }
  rp_sort_indices_desc_by_score(indices, scores, vocab_size);

  for (size_t i = (size_t)top_k; i < vocab_size; ++i) {
    candidates[indices[i]].keep = 0;
  }

  free(scores);
  free(indices);
}

static int rp_softmax_kept(rp_candidate *candidates, size_t vocab_size) {
  double max_logit = -INFINITY;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (!candidates[i].keep) {
      continue;
    }
    if (candidates[i].logit > max_logit) {
      max_logit = candidates[i].logit;
    }
  }
  if (!isfinite(max_logit)) {
    return 0;
  }

  double sum = 0.0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (!candidates[i].keep) {
      candidates[i].prob = 0.0;
      continue;
    }
    double p = exp(candidates[i].logit - max_logit);
    candidates[i].prob = p;
    sum += p;
  }
  if (sum <= 0.0 || !isfinite(sum)) {
    return 0;
  }
  for (size_t i = 0; i < vocab_size; ++i) {
    if (candidates[i].keep) {
      candidates[i].prob /= sum;
    }
  }
  return 1;
}

static void rp_apply_top_p(rp_candidate *candidates, size_t vocab_size, float top_p) {
  if (top_p >= 1.0f) {
    return;
  }
  if (top_p <= 0.0f) {
    top_p = 1e-6f;
  }

  size_t kept = rp_count_kept(candidates, vocab_size);
  if (kept <= 1) {
    return;
  }

  size_t *indices = (size_t *)malloc(kept * sizeof(size_t));
  if (indices == NULL) {
    return;
  }
  double *scores = (double *)malloc(vocab_size * sizeof(double));
  if (scores == NULL) {
    free(indices);
    return;
  }

  size_t pos = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    scores[i] = candidates[i].prob;
    if (candidates[i].keep) {
      indices[pos++] = i;
    }
  }
  rp_sort_indices_desc_by_score(indices, scores, kept);

  double cumulative = 0.0;
  int kept_any = 0;
  for (size_t i = 0; i < kept; ++i) {
    size_t idx = indices[i];
    cumulative += candidates[idx].prob;
    kept_any = 1;
    if (cumulative >= top_p && i + 1 < kept) {
      for (size_t j = i + 1; j < kept; ++j) {
        candidates[indices[j]].keep = 0;
      }
      break;
    }
  }

  if (!kept_any) {
    candidates[indices[0]].keep = 1;
  }

  free(scores);
  free(indices);
}

static void rp_apply_min_p(rp_candidate *candidates, size_t vocab_size, float min_p) {
  if (min_p <= 0.0f) {
    return;
  }

  double max_prob = 0.0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (candidates[i].keep && candidates[i].prob > max_prob) {
      max_prob = candidates[i].prob;
    }
  }
  if (max_prob <= 0.0f) {
    return;
  }

  const double threshold = (double)min_p * max_prob;
  size_t kept = 0;
  size_t fallback_idx = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (!candidates[i].keep) {
      continue;
    }
    if (candidates[i].prob + 1e-8 < threshold) {
      candidates[i].keep = 0;
    } else {
      fallback_idx = i;
      ++kept;
    }
  }
  if (kept == 0) {
    candidates[fallback_idx].keep = 1;
  }
}

static size_t rp_select_max_logit(const rp_candidate *candidates, size_t vocab_size) {
  double best = -INFINITY;
  size_t best_idx = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (!candidates[i].keep) {
      continue;
    }
    if (candidates[i].logit > best) {
      best = candidates[i].logit;
      best_idx = i;
    }
  }
  return best_idx;
}

static size_t rp_sample_index(rp_sampler_state *state, const rp_candidate *candidates,
                              size_t vocab_size) {
  double threshold = rp_next_uniform(state);
  double cumulative = 0.0;
  size_t last_kept = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (!candidates[i].keep) {
      continue;
    }
    cumulative += candidates[i].prob;
    last_kept = i;
    if (threshold <= cumulative) {
      return i;
    }
  }
  return last_kept;
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
  if (state == NULL || config == NULL || token_ids == NULL || logits == NULL ||
      vocab_size == 0) {
    return 0u;
  }

  rp_candidate *candidates =
      (rp_candidate *)malloc(vocab_size * sizeof(rp_candidate));
  if (candidates == NULL) {
    return token_ids[0];
  }

  size_t effective_history_window = config->history_window;
  if (effective_history_window > sizeof(state->history) / sizeof(state->history[0])) {
    effective_history_window = sizeof(state->history) / sizeof(state->history[0]);
  }
  if (effective_history_window == 0) {
    effective_history_window = sizeof(state->history) / sizeof(state->history[0]);
  }

  for (size_t i = 0; i < vocab_size; ++i) {
    double logit = logits[i];
    const size_t repeat_count =
        rp_count_token_in_history(state, token_ids[i], effective_history_window);
    if (repeat_count > 0) {
      if (config->repeat_penalty > 0.0f && config->repeat_penalty != 1.0f) {
        if (logit >= 0.0f) {
          logit /= config->repeat_penalty;
        } else {
          logit *= config->repeat_penalty;
        }
      }
      logit -= config->presence_penalty;
      logit -= (float)repeat_count * config->frequency_penalty;
    }

    if (config->temperature > 0.0f) {
      logit /= config->temperature;
    }
    candidates[i].token_id = token_ids[i];
    candidates[i].logit = logit;
    candidates[i].prob = 0.0;
    candidates[i].keep = 1;
  }

  rp_apply_top_k(candidates, vocab_size, config->top_k);

  size_t chosen_idx = 0;
  if (config->temperature <= 0.0f) {
    chosen_idx = rp_select_max_logit(candidates, vocab_size);
  } else {
    if (!rp_softmax_kept(candidates, vocab_size)) {
      chosen_idx = rp_select_max_logit(candidates, vocab_size);
    } else {
      rp_apply_top_p(candidates, vocab_size, config->top_p);
      if (!rp_softmax_kept(candidates, vocab_size)) {
        chosen_idx = rp_select_max_logit(candidates, vocab_size);
      } else {
        rp_apply_min_p(candidates, vocab_size, config->min_p);
        if (!rp_softmax_kept(candidates, vocab_size)) {
          chosen_idx = rp_select_max_logit(candidates, vocab_size);
        } else {
          chosen_idx = rp_sample_index(state, candidates, vocab_size);
        }
      }
    }
  }

  uint32_t sampled = candidates[chosen_idx].token_id;
  free(candidates);

  if (state->history_len < sizeof(state->history) / sizeof(state->history[0])) {
    state->history[state->history_len++] = sampled;
  } else {
    memmove(state->history, state->history + 1,
            (sizeof(state->history) / sizeof(state->history[0]) - 1) *
                sizeof(state->history[0]));
    state->history[sizeof(state->history) / sizeof(state->history[0]) - 1] = sampled;
    state->history_len = sizeof(state->history) / sizeof(state->history[0]);
  }

  return sampled;
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
