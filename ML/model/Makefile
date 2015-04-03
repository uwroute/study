GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

LR : logistic.cpp libModel.a
	g++ -g -o LR -O3 logistic.cpp -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lModel -L../opt/ -lopt -L../data/ -ldata

libModel.a : logistic_model.o
	ar r libModel.a logistic_model.o

logistic_model.o : logistic_model.cpp
	g++ -g -c -O3 logistic_model.cpp -I../


.PHONY : clean
clean : 
	-rm -rf LR logistic_model.o libModel.a
