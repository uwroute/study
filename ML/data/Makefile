load_data : data_test.cpp libdata.a
	g++ -o load_data -O3 data_test.cpp -I../ -L./ -ldata

libdata.a : data.o
	ar r libdata.a data.o

data.o : data.cpp
	g++ -c data.cpp -O3 -I../

.PHONY : clean
clean : 
	-rm -rf load_data data.o libdata.a
