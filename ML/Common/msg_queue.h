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

template class MessageQueue<T> {
public:
    T pop() {
        Lock lock(Mutex(_m_mutex));
        T tmp = _queue.front();
        _queue.pop_front();
        return tmp;
    }
    void push(T& elem) {
        Lock lock(Mutex(_m_mutex));
        _queue.push_back(elem);
    }
    size_t size() {
        return _queue.size();
    }
private:
    pthread_mutex_t _m_mutex;
    std::deque<T> _queue;
};

#endif