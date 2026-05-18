#include "random_token.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int_arg(const char *value, int min_value, int *out) {
  char *end = NULL;
  errno = 0;
  long parsed = strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < min_value ||
      parsed > INT_MAX) {
    return 0;
  }
  *out = (int)parsed;
  return 1;
}

static int parse_float_arg(const char *value, float min_value, float *out) {
  char *end = NULL;
  errno = 0;
  float parsed = strtof(value, &end);
  if (errno != 0 || end == value || *end != '\0' || parsed < min_value) {
    return 0;
  }
  *out = parsed;
  return 1;
}

static int parse_float_any_arg(const char *value, float *out) {
  char *end = NULL;
  errno = 0;
  float parsed = strtof(value, &end);
  if (errno != 0 || end == value || *end != '\0') {
    return 0;
  }
  *out = parsed;
  return 1;
}

static int parse_uint_arg(const char *value, uint32_t *out) {
  char *end = NULL;
  errno = 0;
  unsigned long parsed = strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > UINT_MAX) {
    return 0;
  }
  *out = (uint32_t)parsed;
  return 1;
}

static void print_usage(const char *program) {
  fprintf(stderr,
          "Usage: %s [count] [--table]\n"
          "       [--seed N] [--temperature X] [--top-k K] [--top-p P]\n"
          "       [--min-p P] [--repeat-penalty X] [--frequency-penalty X]\n"
          "       [--presence-penalty X] [--history-window N]\n",
          program);
}

int main(int argc, char **argv) {
  int count = 16;
  int count_set = 0;
  int use_table_demo = 0;
  rp_sampling_config config;
  rp_sampler_state state;
  rp_default_sampling_config(&config);

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--table") == 0) {
      use_table_demo = 1;
      continue;
    }
    if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
      if (!parse_uint_arg(argv[++i], &config.seed)) {
        fprintf(stderr, "seed must be a valid uint32\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--temperature") == 0 && i + 1 < argc) {
      if (!parse_float_arg(argv[++i], 0.0f, &config.temperature)) {
        fprintf(stderr, "temperature must be >= 0\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
      if (!parse_int_arg(argv[++i], 0, &config.top_k)) {
        fprintf(stderr, "top-k must be >= 0\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
      if (!parse_float_arg(argv[++i], 0.0f, &config.top_p) || config.top_p > 1.0f) {
        fprintf(stderr, "top-p must be in [0, 1]\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--min-p") == 0 && i + 1 < argc) {
      if (!parse_float_arg(argv[++i], 0.0f, &config.min_p) || config.min_p > 1.0f) {
        fprintf(stderr, "min-p must be in [0, 1]\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--repeat-penalty") == 0 && i + 1 < argc) {
      if (!parse_float_arg(argv[++i], 0.0f, &config.repeat_penalty)) {
        fprintf(stderr, "repeat-penalty must be >= 0\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--frequency-penalty") == 0 && i + 1 < argc) {
      if (!parse_float_any_arg(argv[++i], &config.frequency_penalty)) {
        fprintf(stderr, "frequency-penalty must be a valid float\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--presence-penalty") == 0 && i + 1 < argc) {
      if (!parse_float_any_arg(argv[++i], &config.presence_penalty)) {
        fprintf(stderr, "presence-penalty must be a valid float\n");
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--history-window") == 0 && i + 1 < argc) {
      int parsed_window = 0;
      if (!parse_int_arg(argv[++i], 0, &parsed_window)) {
        fprintf(stderr, "history-window must be >= 0\n");
        return 1;
      }
      config.history_window = (size_t)parsed_window;
      continue;
    }

    if (count_set) {
      print_usage(argv[0]);
      return 1;
    }
    if (!parse_int_arg(arg, 1, &count)) {
      print_usage(argv[0]);
      return 1;
    }
    count_set = 1;
  }

  if (use_table_demo) {
    rp_reset();
    for (int i = 0; i < count; ++i) {
      printf("%u\n", rp_next_token_id());
    }
    return 0;
  }

  rp_sampler_init(&state, &config);
  const uint32_t *token_ids = rp_offline_logits_token_ids();
  for (int i = 0; i < count; ++i) {
    const float *step_logits = rp_offline_logits_for_step((size_t)i);
    uint32_t token =
        rp_sample_from_logits(&state, &config, token_ids, step_logits,
                              rp_offline_logits_vocab_size());
    printf("%u\n", token);
  }

  return 0;
}
