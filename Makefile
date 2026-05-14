# Compilers (override with `make MPICXX=... NVCC=...`)
MPICXX  ?= mpicxx
NVCC    ?= nvcc

# Flags
CXXFLAGS  ?= -O2 -std=c++11 -I./src -I.
NVCCFLAGS ?= -O2 -std=c++11 -I./src -I.

# Sources
GATAV_SRC       := src/gAtav.cpp
GATAVGPU_SRC    := src/gAtav_gpu.cu
GATA_SRC        := src/gAta.cpp
GATAGPU_SRC     := src/gAta_gpu.cu

# Objects
GATAV_OBJ       := src/gAtav.o
GATAVGPU_OBJ    := src/gAtav_gpu.o
GATA_OBJ        := src/gAta.o
GATAGPU_OBJ     := src/gAta_gpu.o

# Examples
GATAV_EX_SRC       := example/gatav_example.cpp
GATAVGPU_EX_SRC    := example/gatav_gpu_example.cu
GATA_EX_SRC        := example/gata_example.cpp
GATAGPU_EX_SRC     := example/gata_gpu_example.cu
GATAV_EX_BIN       := example/gatav_example
GATAVGPU_EX_BIN    := example/gatav_gpu_example
GATA_EX_BIN        := example/gata_example
GATAGPU_EX_BIN     := example/gata_gpu_example

# Headers (for dependency tracking)
HDRS := src/gAta.h gata_common.h

.PHONY: all cpu gpu uniform uniform-gpu clean
all: cpu gpu uniform uniform-gpu
cpu:         $(GATAV_EX_BIN)
gpu:         $(GATAVGPU_EX_BIN)
uniform:     $(GATA_EX_BIN)
uniform-gpu: $(GATAGPU_EX_BIN)

# --- src objects -----------------------------------------------------------

$(GATAV_OBJ): $(GATAV_SRC) $(HDRS)
	$(MPICXX) $(CXXFLAGS) -c $(GATAV_SRC) -o $@

$(GATA_OBJ): $(GATA_SRC) $(HDRS)
	$(MPICXX) $(CXXFLAGS) -c $(GATA_SRC) -o $@

# nvcc with -ccbin mpicxx so MPI headers/libs are picked up automatically
$(GATAVGPU_OBJ): $(GATAVGPU_SRC) $(HDRS)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -c $(GATAVGPU_SRC) -o $@

$(GATAGPU_OBJ): $(GATAGPU_SRC) $(HDRS)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) -c $(GATAGPU_SRC) -o $@

# --- examples --------------------------------------------------------------

$(GATAV_EX_BIN): $(GATAV_EX_SRC) $(GATAV_OBJ) $(HDRS)
	$(MPICXX) $(CXXFLAGS) $(GATAV_EX_SRC) $(GATAV_OBJ) -o $@

$(GATA_EX_BIN): $(GATA_EX_SRC) $(GATA_OBJ) $(HDRS)
	$(MPICXX) $(CXXFLAGS) $(GATA_EX_SRC) $(GATA_OBJ) -o $@

$(GATAVGPU_EX_BIN): $(GATAVGPU_EX_SRC) $(GATAV_OBJ) $(GATAVGPU_OBJ) $(HDRS)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) $(GATAVGPU_EX_SRC) $(GATAV_OBJ) $(GATAVGPU_OBJ) -o $@

$(GATAGPU_EX_BIN): $(GATAGPU_EX_SRC) $(GATA_OBJ) $(GATAGPU_OBJ) $(HDRS)
	$(NVCC) -ccbin $(MPICXX) $(NVCCFLAGS) $(GATAGPU_EX_SRC) $(GATA_OBJ) $(GATAGPU_OBJ) -o $@

clean:
	rm -f $(GATAV_OBJ) $(GATAVGPU_OBJ) $(GATA_OBJ) $(GATAGPU_OBJ) \
	      $(GATAV_EX_BIN) $(GATAVGPU_EX_BIN) $(GATA_EX_BIN) $(GATAGPU_EX_BIN)
