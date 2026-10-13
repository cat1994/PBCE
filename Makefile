BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= 4
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3

.PHONY: all configure build pbce test figures verify preflight release help
all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

pbce: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS) --target run_simultaneous_ltbr_pbce

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

figures:
	$(PYTHON) bin/plot_PBCE.py

verify:
	$(PYTHON) scripts/verify_release.py

preflight:
	$(PYTHON) scripts/check_git_release.py

release:
	$(PYTHON) scripts/export_release.py

help:
	@echo "make [JOBS=4]    Build the standalone runner and tests"
	@echo "make test        Run C++ tests, Python tests and integration smoke"
	@echo "make figures     Redraw fixed/adaptive paper figures from bundled data"
	@echo "make verify      Check publication files and readable bundled data"
	@echo "make preflight   Check the candidate Git release without staging or pushing"
	@echo "make release     Optional ZIP export (not required for Git publication)"
