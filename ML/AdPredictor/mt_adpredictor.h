/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-10 18:10
#
# Filename: mt_adpredictor.h
#
# Description: mt_adpredictor
#
=============================================================================*/

#ifndef __MT_ADPREDICTOR_H__
#define __MT_ADPREDICTOR_H__

#include <map>
// #include <tr1/unordered_map>
#include <string>
#include <deque>
#include <stdint.h>
#include <unordered_map>
#inlcude "adpredictor.h"
#include "data/data.h"
#include "Common/thread.h"
#include "Common/lock.h"

namespace ML
{
struct Sample
{
    LongFeature* x;
    double y;
};

struct ThreadData {
    Adpredictor* model;
    MessageQueue<Sample>* queue;
};

class AdPredictorThread : public Thread {
public:
    virtual void run();
    void set(ThreadData& data) {_data=data;}
private:
    ThreadData _data;
};

class ParallelAdPredictor {
public:
    typedef std::unordered_map<uint64_t, double> DoubleHashMap;
public:
    explicit ParallelAdPredictor(int tn) : _thread_num(tn), _slave_models(_thread_num) {}
    ~ParallelAdPredictor() {}
    void init(double mean, double variance, double beta, double eps, size_t max_fea_num = 1000*10000, bool ues_bias=true, double bias=1.0);
    void train(LongFeature* sample, double label);
    void save_model(const std::string& file);
    void load_model(const std::string& file);
private:
    Adpredictor _primary_model;
    std::vector<Adpredictor> _slave_models;
    std::vector<MessageQueue<Sample> >_queue;
    int _thread_num;
};

}
#endif