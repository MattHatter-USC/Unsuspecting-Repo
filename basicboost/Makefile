COMPILER_VERSION := "$(CXX)-$(shell $(CXX) --version | head -n1 | cut -d' ' -f4)"
BUILD_HOST:=$(shell sh -c './BUILD-HOST-GEN')

CFLAGS =  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include -mmic -openmp -openmp-offload -vec-report=1 -DCOMPILER_VERSION=\"$(COMPILER_VERSION)\" -DBUILD_HOST=\"$(BUILD_HOST)\"  -L${MKLROOT}/lib/mic -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm

cell_clustering: cell_clustering.cpp util.cpp util.hpp Makefile
	$(CXX) -o $@ cell_clustering.cpp util.cpp $(CFLAGS) -Wall -lrt -tbb

clean:
	rm -rf cell_clustering
