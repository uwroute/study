GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

SparseLR : SparseLR_test.cpp libSparseLR.a
	g++ -g -o SparseLR SparseLR_test.cpp -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lSparseLR -I../../ -L../Common/ -lCommon -O3

libSparseLR.a : SparseBatchLR.o
	ar r libSparseLR.a SparseBatchLR.o

SparseBatchLR.o : SparseBatchLR.cpp
	g++ -g -c -O3 SparseBatchLR.cpp -I../../


.PHONY : clean
clean : 
	-rm -rf SparseLR_test.o SparseLR SparseBatchLR.o libSparseLR.a
