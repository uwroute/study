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

#include "adpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include <system.h>
#include <time.h>
#include <random.h>

namespace ML {

void AdpredictorThread::run()
{
    Sample sample = NULL;
    bool is_finish = false;
    while (!is_finish)
    {
            if (_data.queue.size() > 0)
            {
                sample = _data.queue.pop();
                if (sample)
                {
                    _data.adpredictor.train(sample.x, sample.y);
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
    for (size_t i=0; i<_slave_models.size(); ++i)
    {
        _slave_models[i].init(mean, variance, beta, eps, max_fea_num, use_bias, bias);
    }
}

void ParallelAdPredictor::train(DataSet& set)
{

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