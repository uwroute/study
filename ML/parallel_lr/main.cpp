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

DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "ftrl.model", "model file");
DEFINE_double(lamda1, 0.03, "L1");
DEFINE_double(lamda2, 1.0, "L2");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

namespace ML {

OptState opt_state;
ReadThreadStatus read_state;
GradThreadStatus grad_status;
ParamSet param;
int GRAD_THREAD_NUM = 0;

}

using namespace ML;

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
