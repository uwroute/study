#GFLAGS_INCLUDE_PATH=../../Common/ThirdParty/gflags/include/
#GFLAGS_LIB_PATH=../../Common/ThirdParty/gflags/lib/

#CXXFLAGS=-I${GFLAGS_INCLUDE_PATH}
#LDFLAGS=-L${GFLAGS_LIB_PATH}
#LIBS=-lgflags

libopt.a : owlqn.o
	ar r libopt.a owlqn.o

owlqn.o : owlqn.cpp
	g++ -g -c owlqn.cpp -O3 -I../

.PHONY : clean
clean : 
	-rm -rf owlqn.o libopt.a
