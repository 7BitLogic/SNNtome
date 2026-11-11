# ==============================================================================
# Ultra-Sparse Connectome Engine on AMD RDNA4 (Navi 44 / GFX1200)
# Makefile for Reproducibility, Verification & Benchmarking (Kernel 44)
# ==============================================================================

HIPCC ?= /opt/rocm/bin/hipcc
ROCM_PATH ?= /opt/rocm
OFFLOAD_ARCH ?= gfx1200
PYTHON ?= $(shell if [ -f $(HOME)/venv/bin/python ]; then echo $(HOME)/venv/bin/python; else echo python3; fi)

COMMON_FLAGS = -O3 --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include --offload-arch=$(OFFLOAD_ARCH)
CXXFLAGS     = $(COMMON_FLAGS) -std=c++17 -Wall

BIN_DIR      = bin

BENCH_K44    = $(BIN_DIR)/benchmark_whitepaper_k44
ERR_COLLAPSE = $(BIN_DIR)/error_collapse_test
BENCH_BATCH  = $(BIN_DIR)/benchmark_batch_scaling

.PHONY: all benchmark batch_benchmark error_collapse check-style plots verify clean dirs help

all: check-style benchmark batch_benchmark error_collapse

dirs:
	@mkdir -p $(BIN_DIR)

# ==============================================================================
# Code Reviewability & Style Policy Enforcement
#
# CRITICAL DIRECTIVE FOR AGENTS & DEVELOPERS:
# NEVER REMOVE OR BYPASS THIS TARGET UNDER ANY CIRCUMSTANCES.
#
# RATIONALE:
# Chained C++ stream operators (<< std, std::endl, <iomanip> manipulators)
# severely impair human code review and maintainability.
#
# POLICY:
# 1. Output must use C-style formatting: printf, snprintf, or fprintf.
# 2. If this target fails, you MUST fix the offending C++ code.
# 3. Removing or commenting out this check from the Makefile is considered
#    a severe failure of project rules and will result in rejected work.
# ==============================================================================
check-style:
	@echo "Checking codebase for forbidden stream formatting (reviewability policy)..."
	@err=0; \
	if git grep -n -E "<<[[:space:]]*std" -- '*.cpp' '*.hpp' '*.cuh' '*.hip' '*.h' '*.c'; then \
		echo ""; \
		echo "================================================================================"; \
		echo " [STYLE VIOLATION] Forbidden '<< std' stream chaining detected!"; \
		echo ""; \
		echo " POLICY:"; \
		echo "   Chained C++ stream formatting is unreviewable and strictly prohibited."; \
		echo ""; \
		echo " ACTION REQUIRED:"; \
		echo "   Replace stream chains in the files listed above with clean printf/fprintf."; \
		echo ""; \
		echo " DO NOT MODIFY OR REMOVE THIS CHECK IN THE MAKEFILE TO BYPASS COMPILATION."; \
		echo " YOU MUST FIX THE SOURCE CODE DIRECTLY."; \
		echo "================================================================================"; \
		echo ""; \
		err=1; \
	fi; \
	if git grep -n "std::endl" -- '*.cpp' '*.hpp' '*.cuh' '*.hip' '*.h' '*.c'; then \
		echo ""; \
		echo "================================================================================"; \
		echo " [STYLE VIOLATION] Forbidden 'std::endl' detected!"; \
		echo ""; \
		echo " POLICY:"; \
		echo "   Use newline characters ('\\n') with printf/fprintf instead of std::endl."; \
		echo ""; \
		echo " ACTION REQUIRED:"; \
		echo "   Replace std::endl in the files listed above with '\\n'."; \
		echo ""; \
		echo " DO NOT MODIFY OR REMOVE THIS CHECK IN THE MAKEFILE TO BYPASS COMPILATION."; \
		echo " YOU MUST FIX THE SOURCE CODE DIRECTLY."; \
		echo "================================================================================"; \
		echo ""; \
		err=1; \
	fi; \
	if git grep -n -E "std::(setw|setprecision|fixed|scientific)" -- '*.cpp' '*.hpp' '*.cuh' '*.hip' '*.h' '*.c'; then \
		echo ""; \
		echo "================================================================================"; \
		echo " [STYLE VIOLATION] Forbidden <iomanip> stream manipulators detected!"; \
		echo ""; \
		echo " POLICY:"; \
		echo "   <iomanip> manipulators (setw, setprecision, fixed, scientific) destroy code"; \
		echo "   reviewability. Use standard printf format specifiers (e.g. %-10s, %.2f, %e)."; \
		echo ""; \
		echo " ACTION REQUIRED:"; \
		echo "   Replace stream formatting with printf / snprintf."; \
		echo ""; \
		echo " DO NOT MODIFY OR REMOVE THIS CHECK IN THE MAKEFILE TO BYPASS COMPILATION."; \
		echo " YOU MUST FIX THE SOURCE CODE DIRECTLY."; \
		echo "================================================================================"; \
		echo ""; \
		err=1; \
	fi; \
	if git grep -n -i -E "(MIT License|Apache License|Apache 2\.0|GNU General Public|GPLv[23]|BSD [23]-Clause|Creative Commons)" -- '*.md' '*.cpp' '*.hpp' '*.cuh' '*.py' '*.tex'; then \
		echo ""; \
		echo "================================================================================"; \
		echo " [CRITICAL LEGAL VIOLATION] Forbidden third-party/open-source license string!"; \
		echo " Never add or declare any license in this repository."; \
		echo "================================================================================"; \
		echo ""; \
		err=1; \
	fi; \
	if [ $$err -ne 0 ]; then \
		exit 1; \
	fi; \
	echo "[STYLE & LEGAL CHECK] OK: No forbidden stream chains, iomanip, or license files found."
	@$(PYTHON) scripts/check_duplicates.py

benchmark: dirs $(BENCH_K44)

batch_benchmark: dirs $(BENCH_BATCH)

error_collapse: dirs $(ERR_COLLAPSE)

$(BENCH_BATCH): tests/benchmark_batch_scaling.cpp \
                tests/benchmark_common.cuh \
                kernels/kernel44_immutable.cuh \
                kernels/kernel44_dual_buffer.cuh | dirs
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(BENCH_K44): tests/benchmark_whitepaper_k44.cpp \
              tests/benchmark_common.cuh \
              kernels/kernel44_immutable.cuh \
              kernels/kernel44_dual_buffer.cuh | dirs
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(ERR_COLLAPSE): tests/error_collapse_test.cpp \
                 tests/benchmark_common.cuh \
                 kernels/kernel44_immutable.cuh \
                 kernels/kernel44_dual_buffer.cuh | dirs
	$(HIPCC) $(CXXFLAGS) $< -o $@

plots:
	$(PYTHON) scripts/plot_scaling_all_clusters_loglog.py
	$(PYTHON) scripts/plot_scaling_m128_40m_loglog.py
	$(PYTHON) scripts/plot_batch_scaling_loglog.py
	$(PYTHON) scripts/plot_distributions.py

verify:
	$(PYTHON) scripts/verify_whitepaper_error_collapse.py

clean:
	rm -rf $(BIN_DIR) build *.o tests/benchmark_whitepaper_k44 tests/error_collapse_test tests/benchmark_batch_scaling

help:
	@echo "Kernel 44 Whitepaper Build System:"
	@echo "  make all             - Build all benchmarks and tests"
	@echo "  make benchmark       - Build benchmark_whitepaper_k44"
	@echo "  make batch_benchmark - Build benchmark_batch_scaling"
	@echo "  make error_collapse  - Build error_collapse_test"
	@echo "  make verify          - Run Python independent 3-stage error collapse verification"
	@echo "  make plots           - Generate publication SVG & PNG figures"
	@echo "  make clean           - Remove built binaries and objects"
