SRC = $(wildcard src/*.cu)
OBJ = $(SRC:src/%.cu=build/%.obj)
FLAGS = -O3

.PHONY: clean

bin/program: $(OBJ) | bin
	nvcc $^ -o $@ $(FLAGS)

build/%.obj: src/%.cu | build
	nvcc -c $< -o $@ $(FLAGS)

bin:
	mkdir bin

build:
	mkdir build

clean:
ifeq ($(OS), Windows_NT)
	del /Q bin build
	rmdir bin build
else
	rm -rf bin build
endif
