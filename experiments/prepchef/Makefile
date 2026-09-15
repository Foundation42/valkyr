CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -march=native -Wall -Wextra -Wno-unused-parameter

all: build/prepchef build/din2vtr

build/prepchef: src/experiment.cpp src/common.hpp src/context.hpp src/learner.hpp \
                src/predictor.hpp src/engine.hpp src/scorer.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ src/experiment.cpp

build/din2vtr: src/din2vtr.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ src/din2vtr.cpp

clean:
	rm -rf build

.PHONY: all clean
