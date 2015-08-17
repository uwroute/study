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

struct RWMutex {
public:
    RWMutex(pthread_rwlock_t& m);
    ~RWMutex();
    void rlock();
    void wlock();
    void unlock();
private:
    pthread_rwlock_t* _p_m_mutex;
};

struct  RLock
{
public:
    RLock(RWMutex& m);
    ~RLock();
private:
    RWMutex* _p_mutex;
};

struct  WLock
{
public:
    WLock(RWMutex& m);
    ~WLock();
private:
    RWMutex* _p_mutex;
};

}
#endif