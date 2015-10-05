WORK_PATH=../../../
GFLAGS_INCLUDE_PATH=${WORK_PATH}/ThirdParty/gflags/include/
GFLAGS_LIB_PATH=${WORK_PATH}/ThirdParty/gflags/lib/

CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
LDFLAGS=-L${GFLAGS_LIB_PATH}
LIBS=-lgflags

libCommon.a : string_util.o log.o lock.o thread.o
	ar r libCommon.a string_util.o log.o lock.o thread.o

string_util.o : string_util.cpp
	g++ -g -c -O3 string_util.cpp

log.o : log.cpp
	g++ -g -c -O3 log.cpp

lock.o: lock.cpp
	g++ -g -c -O3 lock.cpp

thread.o: thread.cpp
	g++ -g -c -O3 thread.cpp -lpthread

.PHONY : clean
clean : 
	-rm -rf libCommon.a string_util.o thread.o log.o lock.o
