/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-06 22:28
#
# Filename: state.h
#
# Description: 
#
=============================================================================*/
#ifndef _STATE_H_
#define _STATE_H_

#include "Common/lock.h"

enum GradThreadState {
	CALC_IDLE = 0，
    CALC_GRAD,
    CALC_LOSS,
    CALC_GRAD_AND_LOSS
};

enum OptState {
	OPT_COMPUTE_DIR = 0,
	OPT_LINEAR_SEARCH,
	OPT_DONE
};

enum ReadThreadState {
	READ_IDLE = 0,
	READ_START,
	READ_DOING,
	READ_END,
};

struct GradThreadStatus
{
public:
	GradThreadStatus() : _mutex(_m_mutex), _state(CALC_IDLE) {}
	int done_num() {return _done_num;}
	GradThreadState get_state() {return _state;}
	void add_done_num() {_done_num++;}
	void init_done_num() {_done_num=0;}
	void set_state(GradThreadState state) {_state = state;}
private:
	GradThreadState _state;
	int _done_num;
	pthread_mutex_t _m_mutex;
    Mutex _mutex;
};

struct ReadThreadStatus
{
public:
	ReadThreadStatus() : _mutex(_m_mutex), _state(READ_IDLE) {}
	void set_state(ReadThreadState state) {_state = state;}
	ReadThreadState get_state() {return _state;}
private:
	ReadThreadState _state;
	pthread_mutex_t _m_mutex;
	Mutex _mutex;
};

extern OptState opt_status;
extern ReadThreadStatus read_status;
extern GradThreadStatus grad_statue;


#endif  // _STATE_H_