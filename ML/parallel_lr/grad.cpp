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

extern OptState opt_status;
extern ReadThreadStatus read_status;
extern GradThreadStatus grad_statue;
extern ParamSet param;

using std::ifstream;
using std::ofstream;

double GradCalcThread::get_w(int i)
{
    if (grad_status == CALC_GRAD || grad_status == CALC_LOSS || grad_status == CALC_GRAD_AND_LOSS)
    {
        return param.get_w(i);
    }
    return param.get_next_w(i);
}

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
double GradCalcThread::log_loss(const wx, double label)
{
    double value = -1.0*label*wx;
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
     _batch_loss += log_loss(sample, label);
}
void GradCalcThread::calc_grad_and_loss(const Feature* sample, double label)
{
	double wx = wx(sample);
    _batch_loss += log_loss(wx, data.labels[i]);
    double h = predict(wx);
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
	param.update_batch_grad(_batch_grads);
}
void GradCalcThread::update_batch_grad()
{
    param.update_batch_next_grad(_batch_grads);
}
void GradCalcThread::update_loss()
{
    param.update_loss(_batch_loss);
}
void GradCalcThread::update_next_loss()
{
    param.update_next_loss(_batch_loss);
}
void GradCalcThread::clear_state()
{
	_batch_grads.clear();
	_batch_loss = 0.0;
}
void GradCalcThread::process_batch()
{
    switch (grad_status)
    {
        case CALC_IDLE:
            break;
        case CALC_GRAD:
            update_batch_grad();
            _batch_grads.clear();
            break;
        case CALC_NEXT_GRAD:
            update_batch_next_grad();
            _batch_grads.clear();
            break;
        case CALC_LOSS:
            update_loss();
            _batch_loss = 0.0;
            break;
        case CALC_NEXT_LOSS:
            update_next_loss();
            _batch_loss = 0.0;
            break;
        case CALC_GRAD_AND_LOSS:
            update_batch_grad();
            update_loss();
            _batch_grads.clear();
            _batch_loss = 0.0;
            break;
        case CALC_NEXT_GRAD_AND_LOSS:
            update_batch_next_grad();
            update_next_loss();
            _batch_grads.clear();
            _batch_loss = 0.0;
            break;
    }
}
void GradCalcThread::run()
{
    while (opt_status != OPT_DONE) {
        // wait for start cond
        // rcv sample and gard calc
        while (!_queue->empty() && read_state != READ_DONE)
        {
            if (_queue->empty())
            {
                usleep(1);
            }
            else
            {
                Sample cur_sample = _queue->pop();
                switch (grad_status)
                {
                    case CALC_IDLE:
                        break;
                    case CALC_GRAD:
                    case CALC_NEXT_GRAD:
                        calc_grad(cur_sample.x, cur_sample.y);
                        break;
                    case CALC_LOSS:
                    case CALC_NEXT_LOSS:
                        calc_loss(cur_sample.x, cur_sample.y);
                        break;
                    case CALC_GRAD_AND_LOSS:
                    case CALC_NEXT_GRAD_AND_LOSS:
                        calc_grad_and_loss(cur_sample.x, cur_sample.y);
                        break;
                }
                _calc_num++;
                if (_calc_num >= _batch)
                {
                    _calc_num = 0;
                    process_batch();
                }
            }
        }
        if (_calc_num > 0)
        {
            _calc_num = 0;
            process_batch();
        }
        grad_status.add_done_num();
    }
}

}