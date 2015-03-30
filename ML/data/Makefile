load_data : data_test.cpp libdata.a
	g++ -o load_data -O data_test.cpp -I../ -L./ -ldata

libdata.a : data.o
	ar r libdata.a data.o

data.o : data.cpp
	g++ -c data.cpp -O -I../

.PHONY : clean
clean : 
	-rm -rf load_data data.o libdata.a
