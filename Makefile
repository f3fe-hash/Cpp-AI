CXX := g++

WARN     := -Wall -Wextra -Wpedantic -Wno-ignored-qualifiers
OPTS     := -O3 -funroll-loops
CXXFLAGS := $(WARN) $(OPTS) -std=c++17

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build

TARGET := $(BUILD_DIR)/ai

SRC := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(SRC))
DIR := $(sort $(dir $(OBJ)))

RED    := \033[31m
YELLOW := \033[33m
GREEN  := \033[32m
BLUE   := \033[34m
RESET  := \033[0m

all: $(TARGET)

$(TARGET): $(OBJ)
	@printf "$(BLUE)  LD     Linking $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET)

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(DIR)
	@printf "$(GREEN)  CXX    Building object $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(DIR):
	@mkdir -p $(DIR)

clean:
	@printf "$(RED)  RM     Building directory $(BUILD_DIR)/\n$(RESET)"
	@rm -rf $(BUILD_DIR)

run:
	@./$(TARGET)

size:
	@wc -c $(TARGET)
