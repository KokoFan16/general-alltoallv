# Compilers (override with `make MPICXX=... NVCC=...`)
MPICXX  ?= mpicxx
NVCC    ?= nvcc

# Flags
CXXFLAGS  ?= -O2 -std=c++11 -I./src -I.
NVCCFLAGS ?= -O2 -std=c++11 -I./src -I.

# All build artifacts (objects + binaries) land here
BUILD_DIR := build

# Sources
GATAV_SRC       := src/gAtav.cpp
GATAVGPU_SRC    := src/gAtav_gpu.cu
GATA_SRC        := src/gAta.cpp
GATAGPU_SRC     := src/gAta_gpu.cu

# Objects (in $(BUILD_DIR))
GATAV_OBJ       := $(BUILD_DIR)/gAtav.o
GATAVGPU_OBJ    := $(BUILD_DIR)/gAtav_gpu.o
GATA_OBJ        := $(BUILD_DIR)/gAta.o
GATAGPU_OBJ     := $(BUILD_DIR)/gAta_gpu.o

# Benchmark _ata sources (uniform alltoall reference algorithms)
ATA_BENCH_SRC := \
	benchmarks/OpenMPI_basic_linear_ata.cpp \
	benchmarks/OpenMPI_pairwise_ata.cpp \
	benchmarks/MPICH_scattered_ata.cpp \
	benchmarks/spreadout_ata.cpp \
	benchmarks/bruck_ata.cpp
ATA_BENCH_OBJ := $(patsubst benchmarks/%.cpp,$(BUILD_DIR)/%.o,$(ATA_BENCH_SRC))

# Benchmark _atav sources (non-uniform alltoallv reference algorithms)
ATAV_BENCH_SRC := \
	benchmarks/OpenMPI_basic_linear_atav.cpp \
	benchmarks/OpenMPI_pairwise_atav.cpp \
	benchmarks/MPICH_scattered_atav.cpp \
	benchmarks/spreadout_atav.cpp \
	benchmarks/exclusive_or_atav.cpp \
	benchmarks/bruck_atav.cpp
ATAV_BENCH_OBJ := $(patsubst benchmarks/%.cpp,$(BUILD_DIR)/%.o,$(ATAV_BENCH_SRC))

# Examples
GATAV_EX_SRC       := example/gatav_example.cpp
GATAVGPU_EX_SRC    := example/gatav_gpu_example.cu
GATA_EX_SRC        := example/gata_example.cpp
GATAGPU_EX_SRC     := example/gata_gpu_example.cu

# Binaries (in $(BUILD_DIR))
GATAV_EX_BIN       := $(BUILD_DIR)/gatav_example
GATAVGPU_EX_BIN    := $(BUILD_DIR)/gatav_gpu_example
GATA_EX_BIN        := $(BUILD_DIR)/gata_example
GATAGPU_EX_BIN     := $(BUILD_DIR)/gata_gpu_example

# Headers (for dependency tracking)
HDRS := src/gAta.h gata_common.h

.PHONY: all ata atav ata-gpu atav-gpu clean
all: ata atav ata-gpu atav-gpu
ata:      $(GATA_EX_BIN)
atav:     $(GATAV_EX_BIN)
ata-gpu:  $(GATAGPU_EX_BIN)
atav-gpu: $(GATAVGPU_EX_BIN)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# --- src objects -----------------------------------------------------------

$(GATAV_OBJ): $(GATAV_SRC) $(HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c $(GATAV_SRC) -o $@

$(GATA_OBJ): $(GATA_SRC) $(HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c $(GATA_SRC) -o $@

# benchmark _ata objects (pattern: build/<name>.o from benchmarks/<name>.cpp)
$(BUILD_DIR)/%.o: benchmarks/%.cpp $(HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# nvcc with -ccbin mpicxx so MPI headers/libs are picked up automatically
$(GATAVGPU_OBJ): $(GATAVGPU_SRC) $(HDRS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -c $(GATAVGPU_SRC) -o $@

$(GATAGPU_OBJ): $(GATAGPU_SRC) $(HDRS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -c $(GATAGPU_SRC) -o $@

# --- examples --------------------------------------------------------------

$(GATAV_EX_BIN): $(GATAV_EX_SRC) $(GATAV_OBJ) $(ATAV_BENCH_OBJ) $(HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(GATAV_EX_SRC) $(GATAV_OBJ) $(ATAV_BENCH_OBJ) -o $@

$(GATA_EX_BIN): $(GATA_EX_SRC) $(GATA_OBJ) $(ATA_BENCH_OBJ) $(HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(GATA_EX_SRC) $(GATA_OBJ) $(ATA_BENCH_OBJ) -o $@

$(GATAVGPU_EX_BIN): $(GATAVGPU_EX_SRC) $(GATAV_OBJ) $(GATAVGPU_OBJ) $(HDRS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) $(GATAVGPU_EX_SRC) $(GATAV_OBJ) $(GATAVGPU_OBJ) -o $@

$(GATAGPU_EX_BIN): $(GATAGPU_EX_SRC) $(GATA_OBJ) $(GATAGPU_OBJ) $(HDRS) | $(BUILD_DIR)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) $(GATAGPU_EX_SRC) $(GATA_OBJ) $(GATAGPU_OBJ) -o $@

clean:
	rm -rf $(BUILD_DIR)
