/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-06 22:37
#
# Filename: main.cpp
#
# Description: 
#
=============================================================================*/

#include "state.h"
#include "data.h"
#include "read.h"
#include "grad.h"
#include "owlqn.h"

OptState opt_state;
ReadThreadState read_state;
GradStatus grad_statue;
ParamSet param;

int main()
{
	// read thread
	ReadThread reader;
	reader.load_data(FLAGS_train_file);
	vector<GradCalcThread> calcers(FLAGS_calc_num);
	OWLQN opt(param);
	reader.start();
	for (size_t i=0; i<calcers.size(); ++i)
	{
		calcers[i].start();
	}
	opt.start();
	reader.join();
	for (size_t i=0; i<calcers.size(); ++i)
	{
		calcers[i].join();
	}
	opt.join();
	opt.save(FLAGS_model_file);
	return 0;	
}