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
#include "data/data.h"
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"
#include "adpredictor.h"

namespace ML
{
struct Sample
{
    LongFeature* x;
    double y;
};

struct ThreadData {
    ThreadData() : model(NULL), queue(NULL), file("") {}
    ~ThreadData() {model=NULL; queue=NULL;}
    AdPredictor* model;
    Common::MessageQueue<Sample>* queue;
    std::string file;
};

class AdPredictorThread : public Common::Thread {
public:
    virtual void run();
    AdPredictorThread() {}
    virtual ~AdPredictorThread() {}
    void set(ThreadData& data) {_data=data;}
private:
    ThreadData _data;
};

class ParallelAdPredictor {
public:
    typedef std::unordered_map<uint64_t, double> DoubleHashMap;
public:
    explicit ParallelAdPredictor(int tn) : _thread_num(tn), _slave_models(_thread_num), _queues(_thread_num), _threads(_thread_num) {}
    ~ParallelAdPredictor() {}
    void init(double mean, double variance, double beta, double eps, size_t max_fea_num = 1000*10000, bool ues_bias=true, double bias=1.0);
    void train(LongFeature* sample, double label);
    double predict(const LongFeature* sample);
    void join();
    void merge_model();
    void save_model(const std::string& file);
    void load_model(const std::string& file);
private:
    AdPredictor _primary_model;
    int _thread_num;
    std::vector<AdPredictor> _slave_models;
    std::vector<Common::MessageQueue<Sample> >_queues;
    std::vector<AdPredictorThread> _threads;
};

}
#endif