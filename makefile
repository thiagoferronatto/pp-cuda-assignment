SRC = $(wildcard *.cu)
OBJ = $(SRC:%.cu=%.obj)
FLAGS = -g

program: $(OBJ)
	nvcc $^ -o $@ $(FLAGS)

%.obj: %.cu
	nvcc -c $< -o $@ $(FLAGS)

clean:
ifeq ($(OS), Windows_NT)
	del /Q *.obj *.exe *.exp *.lib *.pdb *.txt
else
	rm -f *.obj *.exe *.exp *.lib *.pdb *.txt
endif
