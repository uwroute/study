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

RWMutex::RWMutex(pthread_rwlock_t& m) : _p_m_mutex(&m) {
	pthread_rwlock_init(_p_m_mutex, NULL);
}
RWMutex::~RWMutex() {
	pthread_rwlock_destroy(_p_m_mutex);
}
void RWMutex::rlock() { 
	pthread_rwlock_rdlock(_p_m_mutex);
}
void RWMutex::wlock() { 
	pthread_rwlock_wrlock(_p_m_mutex);
}
void RWMutex::unlock() { 
	pthread_rwlock_unlock(_p_m_mutex);
}

RLock::RLock(RWMutex& m) : _p_mutex(&m) {
	_p_mutex->rlock();
}
RLock::~RLock() {
	_p_mutex->unlock();
}

WLock::WLock(RWMutex& m) : _p_mutex(&m) {
	_p_mutex->wlock();
}
WLock::~WLock() {
	_p_mutex->unlock();
}

}