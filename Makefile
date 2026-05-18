CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2
TARGET := random_token_demo
SRC := src/main.c src/random_token.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC)

test: $(TARGET)
	python3 -m unittest discover -s tests -p "test_*.py"

clean:
	rm -f $(TARGET)
