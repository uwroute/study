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

void ThreadData::synchroModel()
{
    {
        Common::Lock lock(*mutex);
        primary_model->update_message(*model);
    }
    model->copy(*primary_model);
}

void AdPredictorThread::run()
{
    Sample sample;
    bool is_finish = false;
    int train_count = 0;
    while (!is_finish)
    {
            if (_data.queue->size() > 0)
            {
                sample = _data.queue->pop();
                if (sample.x != NULL)
                {
                    LOG_DEBUG("rcv sample [%lu:%lf, %lf]", sample.x->index, sample.x->value, sample.y);
                    if ((_data.minBatch != 0) && (train_count == _data.minBatch))
                    {
                        _data.synchroModel();
                    }
                    _data.model->compute_message(sample.x, sample.y);
                    train_count++;
                }
                else
                {
                    _data.synchroModel();
                    is_finish = true;
                    LOG_INFO("Thread %lf end!", sample.y);
                }
            }
            else
            {
                usleep(1);
            }
    }
    LOG_INFO("Train samples count : %d", train_count);
}

void ParallelAdPredictor::init(double mean, double variance, double beta, double eps, int mini_batch, size_t max_fea_num, bool use_bias, double bias)
{
    _primary_model.init(mean, variance, beta, eps, max_fea_num, use_bias, bias);
    _mini_batch = mini_batch;
    for (int i=0; i<_thread_num; ++i)
    {
        LOG_DEBUG("init thread : %d", i+1);
        _slave_models[i].init(mean, variance, beta, eps, max_fea_num, use_bias, bias);
        ThreadData data;
        data.queue = &(_queues[i]);
        data.model = &(_slave_models[i]);
        data.primary_model = &_primary_model;
        data.minBatch = mini_batch;
        data.mutex = &_mutex;
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
        LOG_INFO("merge model : %d", i);
        _primary_model.merge(_slave_models[i]);
    }
}

void ParallelAdPredictor::train(LongFeature* sample, double label)
{
    if (_thread_num == 0)
    {
        if (sample == NULL)
        {
            if (_mini_batch)
            {
                _primary_model.update_message(_primary_model);
            }
            return;
        }
        if (_mini_batch == 0)
        {
            _primary_model.train(sample, label);
        }
        else
        {
            if (_mini_batch && (_mini_batch==_train_count))
            {
                _primary_model.update_message(_primary_model);
                _primary_model.clear_message();
                _train_count = 0;
            }
            _primary_model.compute_message(sample, label);
            _train_count++;
        }
    }
    else
    {
        Sample tmp;
        tmp.x = sample;
        tmp.y = label;
        
        if (tmp.x == NULL) {
            for (int i=0; i<_thread_num; ++i) {
                tmp.y = i*1.0;
                LOG_INFO("push end signal to %d queue", i);
                _queues[i].push(tmp);
            }
        }
        else {
                int idx = rand()%_thread_num;
                while (_queues[idx].size() > 1000) {
                    idx = rand()%_thread_num;
                }
                LOG_DEBUG("push sample [%lu:%lf, %lf] to %d queue", tmp.x->index, tmp.x->value, tmp.y, idx);
                _queues[idx].push(tmp);
        }
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