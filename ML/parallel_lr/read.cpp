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

extern OptState opt_status;
extern ReadThreadStatus read_status;
extern GradThreadStatus grad_statue;

namespace ML{

int ReadThread::load_data(std::string file) {
	return load_data(file, _data);
}

void ReadThread::consume_data() {
	Sample one_sample;
	for (int i=0; i<_data.sample_num; ++i)
	{
		one_sample.x = _data.samples + _data.sample_idx[i];
		one_sample.y = _data.labels[i];
		int rng_queue = rand() % _queues.size();
		_queues[rng_queue].push(one_sample);
	}
}

void ReadThread::run() {
	while (opt_status != OPT_DONE) // opt not done
	{
		// wait for cond of start read
		switch (read_status)
		{
			case READ_IDLE:
				usleep(1);
				break;
			case READ_START:
				read_status.set_state(READ_DOING);
				break;
			case READ_DOING:
				consume_data();
				read_status.set_state(READ_DONE);
				break;
			case READ_DONE:
				usleep(1);
				break;
		}
	}
}

}