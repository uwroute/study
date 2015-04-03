GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

libCommon.a : string_util.o
	ar r libCommon.a string_util.o

string_util.o : string_util.cpp
	g++ -g -c string_util.cpp

.PHONY : clean
clean : 
	-rm -rf libCommon.a string_util.o
