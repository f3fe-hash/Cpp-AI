CXX := g++

ifeq ($(debug),1)
	DEBUG := -DDEBUG
	OPTS := -O0 -g
else
	OPTS := -O3 -funroll-loops -Os
endif

WARN     := -Wall -Wextra -Wpedantic
CXXFLAGS := $(WARN) $(OPTS) $(DEBUG) -std=c++17
LIBS     := -lOpenCL

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build

TARGET := $(BUILD_DIR)/nn

SRC := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(SRC))
DIR := $(sort $(dir $(OBJ)))

RED    := \033[91m
YELLOW := \033[93m
GREEN  := \033[92m
BLUE   := \033[94m
RESET  := \033[0m

all: $(TARGET)

$(TARGET): $(OBJ)
	@printf "$(BLUE)  LD     Linking $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LIBS)
ifeq ($(debug),1)
	@printf "$(YELLOW)  WARN   Warning: Compiling in DEBUG MODE\n"
endif

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(DIR)
	@printf "$(GREEN)  CXX    Building object $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(GPU) -c -o $@ $<

$(DIR):
	@mkdir -p $(DIR)

clean:
	@printf "$(RED)  RM     Building directory $(BUILD_DIR)/\n$(RESET)"
	@rm -rf $(BUILD_DIR)

run:
	@printf "$(YELLOW)  RUN    Running executable $(TARGET)\n$(RESET)"
ifeq ($(debug),1)
	@gdb $(TARGET)
else
	@./$(TARGET)
endif
	@printf "$(YELLOW)  RUN    Done running executable $(TARGET)\n$(RESET)"

size:
	@wc -c < $(TARGET) | awk '{printf "%.2f KB\n", $$1 / 1000}'

