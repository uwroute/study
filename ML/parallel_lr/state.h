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

enum GradThreadState {
    CALC_GRAD = 0,
    CALC_LOSS,
    CALC_GRAD_AND_LOSS,
};

enum OptState {
	COMPUTE_DIR = 0,
	LINEAR_SEARCH,
	OPT_DONE,
};

enum ReadThreadState {
	START_READ = 0,
	IN_READ,
	END_READ
};

struct GradStatus
{
	GradThreadState state;
	int done_num;
};

extern OptState opt_state;
extern ReadThreadState read_state;
extern GradStatus grad_statue;


#endif  // _STATE_H_
