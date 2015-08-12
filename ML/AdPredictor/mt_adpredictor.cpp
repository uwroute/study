/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 15:07
#
# Filename: mt_adpredictor.cpp
#
# Description: mt_adpredictor
#
=============================================================================*/

#include "mt_adpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include <time.h>
#include <unistd.h>

namespace ML {

void AdPredictorThread::run()
{
    Sample sample;
    bool is_finish = false;
    while (!is_finish)
    {
            if (_data.queue->size() > 0)
            {
                sample = _data.queue->pop();
                if (sample.x != NULL)
                {
                    _data.model->train(sample.x, sample.y);
                }
                else
                {
                    is_finish = true;
                }
            }
            else
            {
                  usleep(1);
            }
    }
}

void ParallelAdPredictor::init(double mean, double variance, double beta, double eps, size_t max_fea_num, bool use_bias, double bias)
{
    _primary_model.init(mean, variance, beta, eps, max_fea_num, use_bias, bias);
    for (int i=0; i<_thread_num; ++i)
    {
        _slave_models[i].init(mean, variance, beta, eps, max_fea_num, use_bias, bias);
        ThreadData data;
        data.queue = &(_queues[i]);
        data.model = &(_slave_models[i]);
        _threads[i].set(data);
        _threads[i].start();
    }
}

void ParallelAdPredictor::join()
{
    for (int i=0; i<_thread_num; ++i)
    {
        _threads[i].join();
    }
}

void ParallelAdPredictor::merge_model()
{
    for (int i=0; i<_thread_num; ++i)
    {
        _primary_model.merge(_slave_models[i]);
    }
}

void ParallelAdPredictor::train(LongFeature* sample, double label)
{
    if (_thread_num == 0)
    {
        _primary_model.train(sample, label);
    }
    else
    {
        int idx = rand()%_thread_num;
        Sample tmp;
        tmp.x = sample;
        tmp.y = label;
        _queues[idx].push(tmp);
    }
}

double ParallelAdPredictor::predict(const LongFeature* sample)
{
    return _primary_model.predict(sample);
}

void ParallelAdPredictor::save_model(const std::string& file)
{
    _primary_model.save_model(file);
}

void ParallelAdPredictor::load_model(const std::string& file)
{
    _primary_model.load_model(file);
}

}