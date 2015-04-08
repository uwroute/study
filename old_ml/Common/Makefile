GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

libCommon.a : string_util.o log.o
	ar r libCommon.a string_util.o log.o

string_util.o : string_util.cpp
	g++ -g -c -O3 string_util.cpp

log.o : log.cpp
	g++ -g -c -O3 log.cpp

.PHONY : clean
clean : 
	-rm -rf libCommon.a string_util.o
