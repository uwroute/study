/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 15:42
#
# Filename: thread.cpp
#
# Description: thread
#
=============================================================================*/

#include "thread.h"
#include "log.h"

namespace Common
{
void* Thread::static_run(void* arg) {
      Thread* p = (Thread*)arg;
      p->internal_run();
      return NULL;
}

void* Thread::internal_run() {
       run();
       // pthread_exit(NULL);
       return NULL;
}

int Thread::start() {
	return pthread_create(&_tid, NULL, static_run, this);
}

void Thread::join() {
	if (_tid > 0) {
		LOG_DEBUG("tid : %llu", (uint64_t)_tid);
		pthread_join(_tid, NULL);
	}
}
}
