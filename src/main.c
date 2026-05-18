#include "random_token.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int count = 16;
  if (argc > 1) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || parsed <= 0) {
      fprintf(stderr, "count must be a valid positive integer\n");
      return 1;
    }
    count = (int)parsed;
  }

  rp_reset();
  for (int i = 0; i < count; ++i) {
    printf("%u\n", rp_next_token_id());
  }
  return 0;
}
