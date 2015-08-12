/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 18:29
#
# Filename: lock.cpp
#
# Description: lock
#
=============================================================================*/
#include "lock.h"

namespace Common
{
Mutex::Mutex(pthread_mutex_t m) : _m_mutex(m) {
	pthread_mutex_init(&_m_mutex, NULL);
}
Mutex::~Mutex() {
	pthread_mutex_destroy(&_m_mutex);
}
void Mutex::lock() { 
	pthread_mutex_lock(&_m_mutex);
}
void Mutex::unlock() { 
	pthread_mutex_unlock(&_m_mutex);
}
Lock::Lock(Mutex& m) : _mutex(m) {
	_mutex.lock();
}
Lock::~Lock() {
	_mutex.unlock();
}
}