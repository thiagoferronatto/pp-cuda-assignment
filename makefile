SRC = $(wildcard *.cu)

program: $(SRC)
	nvcc $^ -o $@ -g

clean:
	del /Q *.exe *.exp *.lib *.pdb
