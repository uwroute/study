/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 15:42
#
# Filename: msg_queue.h
#
# Description: msg_queue
#
=============================================================================*/
#ifndef _MESSAGE_QUEUE_H_
#define _MESSAGE_QUEUE_H_

#include "lock.h"
namespace Common
{
template<class T>
class MessageQueue {
public:
    MessageQueue() : _mutex(_m_mutex), _size(0) {}
    T pop() {
        Lock lock(_mutex);
        T tmp = _queue.front();
        _queue.pop_front();
        _size--;
        return tmp;
    }
    void push(T& elem) {
        Lock lock(_mutex);
        _queue.push_back(elem);
        _size++;
    }
    size_t size() {
        return _size;
    }
private:
    pthread_mutex_t _m_mutex;
    Mutex _mutex;
    std::deque<T> _queue;
    size_t _size;
};
// MsgQueue when elem is vector, use swap instead copy
template<class T>
class VectorMessageQueue {
public:
    VectorMessageQueue() : _mutex(_m_mutex), _size(0) {}
    void pop(T& res) {
        Lock lock(_mutex);
        res.swap(_queue.front());
        _queue.pop_front();
        _size--;
        return tmp;
    }
    void push(T& elem) {
        Lock lock(_mutex);
        T e;
        _queue.push_back(e);
        _queue.back().swap(elem);
        _size++;
    }
    size_t size() {
        return _size;
    }
private:
    pthread_mutex_t _m_mutex;
    Mutex _mutex;
    std::deque<T> _queue;
    size_t _size;
};

}

#endif