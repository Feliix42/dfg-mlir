BUILD_DIR=./build
DFG_OPT=../../build/bin/dfg-opt
EXTRACT_ADAPTERS=../../scripts/extract_adapters.py

MLIR_FILES=$(wildcard *.mlir)
TARGETS=$(MLIR_FILES:%.mlir=$(BUILD_DIR)/%.elf)

OTHER_DEP=ll_externs c_externs
SUB_TARGETS=$(OTHER_DEP:%=$(BUILD_DIR)/%.o)

LINKER_FLAGS=-ldpm -lyaml-cpp -lcrypto

all: $(TARGETS)

keep: $(TARGETS:%.elf=%.cpp)

$(BUILD_DIR)/%.elf: $(BUILD_DIR)/%.o $(BUILD_DIR)/%.dpm_calls_o $(SUB_TARGETS)
	@echo Linking $^ into $@
	@clang++ -o $@ $^ $(LINKER_FLAGS) -static
$(BUILD_DIR)/%.o: $(BUILD_DIR)/%.cpp
	@echo Compiling $^ into object file $@
	@clang++ -c $^ -o $@ -static #-O0 -ggdb
$(BUILD_DIR)/%.cpp: $(BUILD_DIR)/%.emitc
	@echo Translating dpm adapter file $^ to c++ file $@
	@mlir-translate --mlir-to-cpp $^ -o $@
$(BUILD_DIR)/%.emitc: %.mlir
	@echo Translating DFG mlir file $^ to dpm adapter file $@
	@$(DFG_OPT) --convert-dfg-to-dpm-wrappers $^ -o $@
$(BUILD_DIR)/%.ll: $(BUILD_DIR)/%.cpp
	@echo Compiling $^ to LLVMIR $@
	@clang++ $^ -S -emit-llvm -o $@ -static
$(BUILD_DIR)/%.dpm_calls_mlir: $(BUILD_DIR)/%.ll %.mlir
	@echo Adding generated DPM function adapter signatures to original MLIR file
	@cp $*.mlir $@
	@python3 $(EXTRACT_ADAPTERS) $< >> $@
$(BUILD_DIR)/%.dpm_calls_mlir_llvm: $(BUILD_DIR)/%.dpm_calls_mlir
	@echo Translating MLIR file with adapter signatures to MLIR in LLVM dialect
	$(DFG_OPT) --convert-dfg-to-dpm-calls --convert-func-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm $^ -o $@
$(BUILD_DIR)/%.dpm_calls_ll: $(BUILD_DIR)/%.dpm_calls_mlir_llvm
	@echo Converting LLVM dialect to LLVMIR
	mlir-translate $^ --mlir-to-llvmir -o $@
$(BUILD_DIR)/%.dpm_calls_o: $(BUILD_DIR)/%.dpm_calls_ll
	llc -filetype=obj $^ -o $@ --relocation-model=pic
$(BUILD_DIR)/%.o: %.ll
	llc -filetype=obj $^ -o $@ --relocation-model=pic
$(BUILD_DIR)/%.o: %.cpp
	clang++ -c $^ -o $@ -static #-O0 -ggdb

.PHONY: clean
clean:
	rm -v $(BUILD_DIR)/*
