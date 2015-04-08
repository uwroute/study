GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

all : ftrl_train ftrl_predict
	echo "all done"

ftrl_train : ftrl_train.cpp libFTRL.a
	g++ -g -o ftrl_train ftrl_train.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lFTRL -L../data/ -ldata -L../Common/ -lCommon

ftrl_predict : ftrl_predict.cpp libFTRL.a
	g++ -g -o ftrl_predict ftrl_predict.cpp -O3 -I../ -I../../Common/ThirdParty/gflags/include/ -L../../Common/ThirdParty/gflags/lib/ -lgflags -L./ -lFTRL -L../data/ -ldata -L../Common/ -lCommon

libFTRL.a : ftrl.o
	ar r libFTRL.a ftrl.o

ftrl.o : ftrl.cpp
	g++ -g -c ftrl.cpp -O3 -I../ -L../Common/ -lCommon -L../data/ -ldata


.PHONY : clean
clean : 
	-rm -rf ftrl.o libFTRL.a ftrl_train ftrl_predict
