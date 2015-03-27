/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-07 09:49
#
# Filename: double_buffer.h
#
# Description: double buffer
#
=============================================================================*/
#ifndef _DOUBLE_BUFFER_H_
#define _DOUBLE_BUFFER_H_

#define DOUBLE_BUFFER(T) \
class T##_Double_Buffer \
{                       \
public:                 \
    T& cur() {return a;} \
    T& cur1() {return flag?a:b;} \
    T& next() {return b;} \
    T& next1() {return !flag?a:b;} \
    void swap() {flag=!flag;} \
private:                \
    T a;                \
    T b;                \
    bool flag;          \
};

#endif  // _DOUBLE_BUFFER_H_
