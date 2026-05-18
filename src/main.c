#include "random_token.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int count = 16;
  if (argc > 1) {
    count = atoi(argv[1]);
    if (count <= 0) {
      fprintf(stderr, "count must be a positive integer\n");
      return 1;
    }
  }

  rp_reset();
  for (int i = 0; i < count; ++i) {
    printf("%u\n", rp_next_token_id());
  }
  return 0;
}
