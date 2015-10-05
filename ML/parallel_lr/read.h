/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 23:10
#
# Filename: read.h
#
# Description: 
#
=============================================================================*/
#ifndef _READ_H_
#define _READ_H_

#include <string>
#include "data.h"
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"

// Read Thread, read global data
// when start compute, send data to muti grad&loss compute thread

namespace ML{

class ReadThread : public Common::Thread {
public:
	ReadThread() {}
	~ReadThread() {}
	void add_queue(SampleQueue* p) {_queues.push_back(p);}
	void load_data(std::string file);
	void consume_data();
	virtual void run();
private:
	DataSet  _data;
	std::vector<SampleQueue*> _queues;
};

}
#endif  // _READ_H_
