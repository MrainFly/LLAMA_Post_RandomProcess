CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Werror -O2
TARGET := random_token_demo
SRC_C := src/main.c src/random_token.c
SRC_CPP := src/sampler_llama.cpp
OBJ := $(SRC_C:.c=.o) $(SRC_CPP:.cpp=.o)
LDLIBS := -lm

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TARGET)
	python3 -m unittest discover -s tests -p "test_*.py"

clean:
	rm -f $(TARGET) $(OBJ)
