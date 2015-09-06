/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:07
#
# Filename: grad.cpp
#
# Description: 
#
=============================================================================*/

#include "logistic_model.h"
#include <cmath>
#include <fstream>
#include "Common/log.h"

namespace ML {

using std::ifstream;
using std::ofstream;

double GradCalcThread::wx(const Feature* sample)
{
    double wx = 0.0;
    while (sample->index != -1)
    {
        wx += sample->value * get_w(sample->index, w);
        sample++;
    }
    return wx;
}
double GradCalcThread::predict(const double wx)
{
    if (wx > 30.0)
    {
        return 1.0;
    }
    else if (wx < -30.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0/(1.0 + exp(-1.0*wx));
    }
}
double GradCalcThread::predict(const Feature* sample)
{
    double value = wx(sample);
    if (value > 30.0)
    {
        return 1.0;
    }
    else if (value < -30.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0/(1.0 + exp(-1.0*value));
    }
}
double GradCalcThread::log_loss(const Feature* sample, double label)
{
    double value = -1.0*label*wx(sample);
    if (value < -30.0)
    {
        return 0.0;
    }
    else if (value > 30.0)
    {
        return value;
    }
    else
    {
        return log(1.0 + exp(value));
    }
}
void GradCalcThread::calc_grad(const Feature* sample, double label)
{
        double h = predict(sample, w);
        double y = data.labels[i];
        if (y < 0.5)
        {
            y = 0.0;
        }
        double g = h - y;
        while (sample->index != -1)
        {
        	if (_batch_grads.find(sample->index) == _batch_grads.end())
        	{
        		_batch_grads[sample->index] = 0;
        	}
            _batch_grads[sample->index] += g*sample->value;
            sample++;
        }
    }
}
void GradCalcThread::calc_loss(const Feature* sample, double label)
{
     _loss += log_loss(sample, label);
}
void GradCalcThread::calc_grad_and_loss(const Feature* sample, double label)
{
	loss += log_loss(sample, w, data.labels[i]);
	double h = predict(sample, w);
	double y = data.labels[i];
	if (y < 0.5)
	{
	    y = 0.0;
	}
	double g = h - y;
	while (sample->index != -1)
	{
	            	if (_batch_grads.find(sample->index) == _batch_grads.end())
	        	{
	        		_batch_grads[sample->index] = 0;
	        	}
	            _batch_grads[sample->index] += g*sample->value;
	            sample++;
	}
}

void GradCalcThread::update_batch_grad()
{
	for (std::unordered_map<int, double>::iterator iter=_batch_grads.begin(); iter != _batch_grads.end(); ++iter)
	{
		update_grad(iter->first, iter->second);
	}
}
void GradCalcThread::clear_state()
{
	_batch_grads.clear();
	_loss = 0.0;
}
void GradCalcThread::run()
{
    while (read_state != OPT_DONE) {
        // wait for start cond
        // rcv sample and gard calc
        while (!_queue.empty() && read_state != READ_DONE)
        {
            if (_queue.empty())
            {
                // sleep
            }
            else
            {
                // grad calc
            }
        }
    }
}

}