#include "random_token.h"

enum {
  RP_GLOBAL_SEED = 0x12345678u
};

static const uint32_t k_offline_token_table[] = {
    128000u, 128001u, 128002u, 128003u, 128004u, 128005u, 128006u, 128007u,
    128008u, 128009u, 128010u, 128011u, 128012u, 128013u, 128014u, 128015u};

static uint32_t g_state = RP_GLOBAL_SEED;

static uint32_t rp_next_state(uint32_t current) {
  return (1103515245u * current + 12345u) & 0x7fffffffu;
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
