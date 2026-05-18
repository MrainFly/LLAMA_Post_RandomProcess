#include "sampler_llama_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct Candidate {
  uint32_t token_id;
  double logit;
  double prob;
};

size_t effective_history_window(const rp_sampling_config *config) {
  if (config->history_window == 0 || config->history_window > RP_MAX_HISTORY) {
    return RP_MAX_HISTORY;
  }
  return config->history_window;
}

size_t token_repeat_count(const rp_sampler_state *state, uint32_t token_id,
                          size_t history_window) {
  size_t count = 0;
  const size_t start =
      state->history_len > history_window ? state->history_len - history_window : 0;
  for (size_t i = start; i < state->history_len; ++i) {
    if (state->history[i] == token_id) {
      ++count;
    }
  }
  return count;
}

void apply_penalties(std::vector<Candidate> &candidates, const rp_sampler_state *state,
                     const rp_sampling_config *config) {
  const size_t window = effective_history_window(config);
  for (auto &c : candidates) {
    const size_t count = token_repeat_count(state, c.token_id, window);
    if (count == 0) {
      continue;
    }

    if (config->repeat_penalty > 0.0f && config->repeat_penalty != 1.0f) {
      c.logit = c.logit <= 0.0 ? c.logit * config->repeat_penalty
                               : c.logit / config->repeat_penalty;
    }
    c.logit -= static_cast<double>(count) * config->frequency_penalty;
    c.logit -= config->presence_penalty;
  }
}

void apply_temperature(std::vector<Candidate> &candidates, float temperature) {
  if (temperature <= 0.0f) {
    return;
  }
  for (auto &c : candidates) {
    c.logit /= temperature;
  }
}

void apply_top_k(std::vector<Candidate> &candidates, int k) {
  if (k <= 0 || static_cast<size_t>(k) >= candidates.size()) {
    return;
  }

  std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                    [](const Candidate &a, const Candidate &b) {
                      if (a.logit == b.logit) {
                        return a.token_id < b.token_id;
                      }
                      return a.logit > b.logit;
                    });
  candidates.resize(static_cast<size_t>(k));
}

void compute_softmax(std::vector<Candidate> &candidates) {
  if (candidates.empty()) {
    return;
  }
  double max_logit = candidates[0].logit;
  for (const auto &c : candidates) {
    max_logit = std::max(max_logit, c.logit);
  }

  double sum = 0.0;
  for (auto &c : candidates) {
    c.prob = std::exp(c.logit - max_logit);
    sum += c.prob;
  }
  if (sum <= 0.0 || !std::isfinite(sum)) {
    const double p = 1.0 / static_cast<double>(candidates.size());
    for (auto &c : candidates) {
      c.prob = p;
    }
    return;
  }
  for (auto &c : candidates) {
    c.prob /= sum;
  }
}

void apply_top_p(std::vector<Candidate> &candidates, float p) {
  if (candidates.size() <= 1 || p >= 1.0f) {
    return;
  }
  const double top_p = p <= 0.0f ? 1e-6 : p;
  compute_softmax(candidates);

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              if (a.prob == b.prob) {
                return a.token_id < b.token_id;
              }
              return a.prob > b.prob;
            });

  double cumulative = 0.0;
  size_t keep = 0;
  for (; keep < candidates.size(); ++keep) {
    cumulative += candidates[keep].prob;
    if (cumulative >= top_p && keep + 1 < candidates.size()) {
      ++keep;
      break;
    }
  }
  candidates.resize(std::max<size_t>(1, keep));
}

void apply_min_p(std::vector<Candidate> &candidates, float min_p) {
  if (candidates.size() <= 1 || min_p <= 0.0f) {
    return;
  }

  double max_logit = -std::numeric_limits<double>::infinity();
  for (const auto &c : candidates) {
    max_logit = std::max(max_logit, c.logit);
  }
  const double threshold = max_logit + std::log(min_p);

  const auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](const Candidate &a, const Candidate &b) {
        if (a.logit == b.logit) {
          return a.token_id > b.token_id;
        }
        return a.logit < b.logit;
      });
  const Candidate fallback = *best_it;

  auto it = std::remove_if(candidates.begin(), candidates.end(),
                           [threshold](const Candidate &c) {
                             return c.logit < threshold;
                           });
  candidates.erase(it, candidates.end());
  if (candidates.empty()) {
    candidates.push_back(fallback);
  }
}

double next_uniform(rp_sampler_state *state) {
  state->rng_state = (1103515245u * state->rng_state + 12345u) & 0x7fffffffu;
  return static_cast<double>(state->rng_state) / static_cast<double>(0x80000000u);
}

uint32_t sample_token(rp_sampler_state *state, const rp_sampling_config *config,
                      std::vector<Candidate> &candidates) {
  if (candidates.empty()) {
    return 0u;
  }
  if (config->temperature <= 0.0f) {
    auto it = std::max_element(candidates.begin(), candidates.end(),
                               [](const Candidate &a, const Candidate &b) {
                                 if (a.logit == b.logit) {
                                   return a.token_id > b.token_id;
                                 }
                                 return a.logit < b.logit;
                               });
    return it->token_id;
  }

  compute_softmax(candidates);
  const double r = next_uniform(state);
  double cumulative = 0.0;
  uint32_t token = candidates.back().token_id;
  for (const auto &c : candidates) {
    cumulative += c.prob;
    token = c.token_id;
    if (r <= cumulative) {
      break;
    }
  }
  return token;
}

void append_history(rp_sampler_state *state, uint32_t token) {
  if (state->history_len < RP_MAX_HISTORY) {
    state->history[state->history_len++] = token;
    return;
  }
  for (size_t i = 1; i < RP_MAX_HISTORY; ++i) {
    state->history[i - 1] = state->history[i];
  }
  state->history[RP_MAX_HISTORY - 1] = token;
  state->history_len = RP_MAX_HISTORY;
}

}  // namespace

extern "C" uint32_t rp_sample_from_logits_llama_cpp(
    rp_sampler_state *state, const rp_sampling_config *config,
    const uint32_t *token_ids, const float *logits, size_t vocab_size) {
  if (state == nullptr || config == nullptr || token_ids == nullptr ||
      logits == nullptr || vocab_size == 0) {
    return 0u;
  }

  std::vector<Candidate> candidates;
  candidates.reserve(vocab_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    candidates.push_back({token_ids[i], logits[i], 0.0});
  }

  apply_penalties(candidates, state, config);
  apply_temperature(candidates, config->temperature);
  apply_top_k(candidates, config->top_k);
  apply_top_p(candidates, config->top_p);
  apply_min_p(candidates, config->min_p);

  const uint32_t sampled = sample_token(state, config, candidates);
  append_history(state, sampled);
  return sampled;
}
