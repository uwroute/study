GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

all : sgd_fm_train sgd_fm_predict
	echo "all done"

sgd_fm_train : sgd_fm_train.cpp libFM.a
	g++ -g -o sgd_fm_train sgd_fm_train.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lFM -L../data/ -ldata -L../Common/ -lCommon

sgd_fm_predict : sgd_fm_predict.cpp libFM.a
	g++ -g -o sgd_fm_predict sgd_fm_predict.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lFM -L../data/ -ldata -L../Common/ -lCommon

libFM.a : sgd_fm.o
	ar r libFM.a sgd_fm.o

sgd_fm.o : sgd_fm.cpp
	g++ -g -c sgd_fm.cpp -O3 -I../ -L../Common/ -lCommon -L../data/ -ldata


.PHONY : clean
clean : 
	-rm -rf sgd_fm.o libFM.a sgd_fm_train sgd_fm_predict
