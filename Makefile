CXX := g++

WARN     := -Wall -Wextra -Wpedantic
OPTS     := -O3 -funroll-loops -Os
CXXFLAGS := $(WARN) $(OPTS) -std=c++17

ifeq ($(gpu),1)
	GPU := -DUSE_CUDA
endif

ifeq ($(gpu),2)
	GPU      := -DUSE_OPENCL
	GPU_LIBS := -lOpenCL
endif

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build

TARGET := $(BUILD_DIR)/ai

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
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(GPU_LIBS)
ifeq ($(gpu),1)
	@printf "$(YELLOW)  WARN   Warning: Using CUDA GPU\n"
endif
ifeq ($(gpu),2)
	@printf "$(YELLOW)  WARN   Warning: Using Intel/AMD GPU\n"
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
	@./$(TARGET)

size:
	@wc -c $(TARGET)
