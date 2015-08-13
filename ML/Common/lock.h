/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 18:29
#
# Filename: lock.h
#
# Description: lock
#
=============================================================================*/
#ifndef _COMMON_LOCK_H_
#define _COMMON_LOCK_H_

#include <pthread.h>
namespace Common
{
struct Mutex {
public:
    Mutex(pthread_mutex_t& m);
    ~Mutex();
    void lock();
    void unlock();
private:
    pthread_mutex_t* _p_m_mutex;
};
struct  Lock
{
public:
    Lock(Mutex& m);
    ~Lock();
private:
    Mutex* _p_mutex;
};
}
#endif