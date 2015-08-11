/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 15:42
#
# Filename: thread.h
#
# Description: thread
#
=============================================================================*/
#ifndef _COMMON_THREAD_H_
#define _COMMON_THREAD_H_

#include <pthread.h>
namespace Common
{
class Thread {
public:
    Thread() {}
    virtual ~Thread() {}
    static void* static_run(void* arg);
public:
    int start();
    void join();
    virtual void run() = 0;
private:
     void* internal_run();
private:
     pthread_t _tid;
};
}
#endif