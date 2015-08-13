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
Mutex::Mutex(pthread_mutex_t& m) : _p_m_mutex(&m) {
	pthread_mutex_init(_p_m_mutex, NULL);
}
Mutex::~Mutex() {
	pthread_mutex_destroy(_p_m_mutex);
}
void Mutex::lock() { 
	pthread_mutex_lock(_p_m_mutex);
}
void Mutex::unlock() { 
	pthread_mutex_unlock(_p_m_mutex);
}
Lock::Lock(Mutex& m) : _p_mutex(&m) {
	_p_mutex->lock();
}
Lock::~Lock() {
	_p_mutex->unlock();
}
}