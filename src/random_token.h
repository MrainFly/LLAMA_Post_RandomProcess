#ifndef RANDOM_TOKEN_H
#define RANDOM_TOKEN_H

#include <stddef.h>
#include <stdint.h>

void rp_reset(void);
uint32_t rp_next_token_id(void);
size_t rp_token_count(void);
const uint32_t *rp_token_table(void);

#endif
