BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= 4
CMAKE ?= cmake
CTEST ?= ctest

.PHONY: all configure build pbce test clean rebuild help

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

pbce: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS) \
		--target run_simultaneous_ltbr_pbce

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)

rebuild: clean build

help:
	@echo "make              Configure and incrementally build all targets"
	@echo "make pbce         Build only the PBCE experiment executable"
	@echo "make test         Build and run all tests"
	@echo "make clean        Remove the selected build directory"
	@echo "make rebuild      Clean and rebuild all targets"
	@echo "make JOBS=8       Build with eight parallel jobs"
	@echo "make BUILD_DIR=build-debug BUILD_TYPE=Debug"
