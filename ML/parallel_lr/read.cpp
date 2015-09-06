/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 23:10
#
# Filename: read.cpp
#
# Description: 
#
=============================================================================*/

#include "read.h"
#include "state.h"

namespace ML{

int ReadThread::load_data(std::string file) {
	return load_data(file, _data);
}

void ReadThread::run() {
	while (read_state != OPT_DONE) // opt not done
	{
		// wait for cond of start read
		// trans data
		read_state = END_READ;
	}
}

}