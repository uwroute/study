GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

all : lr_train lr_predict LR
	echo "all done"

lr_train : lr_train.cpp libModel.a
	g++ -g -o lr_train lr_train.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lModel -L../opt/ -lopt -L../data/ -ldata -L../Common/ -lCommon

lr_predict : lr_predict.cpp libModel.a
	g++ -g -o lr_predict lr_predict.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lModel -L../opt/ -lopt -L../data/ -ldata -L../Common/ -lCommon

LR : logistic.cpp libModel.a
	g++ -g -o LR logistic.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lModel -L../opt/ -lopt -L../data/ -ldata -L../Common/ -lCommon

libModel.a : logistic_model.o
	ar r libModel.a logistic_model.o

logistic_model.o : logistic_model.cpp
	g++ -g -c logistic_model.cpp -O3 -I../ -L../Common/ -lCommon


.PHONY : clean
clean : 
	-rm -rf LR logistic_model.o libModel.a lr_train lr_predict
