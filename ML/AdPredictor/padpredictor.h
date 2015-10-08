/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-17 10:30
#
# Filename: parallel_adpredictor.h
#
# Description: parallel_adpredictor
#
=============================================================================*/

#ifndef __PARALLEL_ADPREDICTOR_H__
#define __PARALLEL_ADPREDICTOR_H__

#include <map>
#include <string>
#include <deque>
#include <stdint.h>
#include <unordered_map>
#include "data/data.h"
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"

namespace ML
{
struct Param {
    Param() : m(0.0), v(0.0) {}
    double m;
    double v;
};

struct Message {
    Message() : vMsg(0.0), mMsg(0.0) {}
    double vMsg;
    double mMsg;
};

struct UpdateRequest {
    std::vector<uint64_t> idxs;
    std::vector<Message> msgs;
};

struct GetRequest {
    std::vector<uint64_t> idxs;
};

struct GetResponse {
    std::vector<uint64_t> idxs;
    std::vector<Message> msgs;
};

// ParameterServer : process update&get from muti client
class ParameterServer {
public:
    typedef std::unordered_map<uint64_t, Message> MessageHashMap;
public:
    ParameterServer() : _rw_mutex(_m_mutex) {}
    ~ParameterServer() {}
public:
    void update(const UpdateRequest& req);
    void get(const GetRequest& req, GetResponse& res);
    void save_params(std::ofstream& out);
private:
    void update(const uint64_t& idx, const Message& msg);
    Message get(const uint64_t& idx);
private:
    MessageHashMap _messages; 
    pthread_rwlock_t _m_mutex;
    Common::RWMutex _rw_mutex;
};

// AdPredictorClient : compute message of every feature in minibatch data
class AdPredictorClient {
public:
    typedef std::unordered_map<uint64_t, Message> MessageHashMap;
    typedef std::unordered_map<uint64_t, Param> ParamHashMap;
public:
    AdPredictorClient() : _bias(1.0),_train_count(0) {}
    ~AdPredictorClient() {}
    void init(double mean, double variance, double beta, double eps, int mini_batch, size_t max_fea_num, bool ues_bias, double down_sample, bool is_update);
    void set_ps(ParameterServer* ps) {_ps = ps;}
    void set_rcv_queue(Common::MessageQueue<LongFeature*>* rcv_queue) {_rcv_queue = rcv_queue;}
    void set_seri(int seri) {_seri = seri;}
    void train_minibatch(LongDataSet& data);
private:
    // set&get
    void update_message(const uint64_t idx, const Message& param);
    Message get_message(uint64_t idx);
    void set_param(const uint64_t idx, const Param& param);
    Param get_param(uint64_t idx);
private:
    // function for train
    void train(LongFeature* sample, double label);
    void active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance);
    double cumulative_probability(double  t, double mean=0.0, double variance=1.0);
    double gauss_probability(double t, double mean=0.0, double variance=1.0);
private:
    // communication with master
    void form_get_request(LongDataSet& data, GetRequest& req);
    void form_update_request(UpdateRequest& req);
private:
    // model param
    MessageHashMap _messages;
    ParamHashMap _params;
    Message _bias_message;
    Param _bias_param;
    Param _prior_param;
    double _beta;
    double _eps;
    // bias config
    uint64_t _bias_idx;
    bool _use_bias;
private:
    // ps train param
    ParameterServer* _ps; // parameter server
    double _down_sample; // down sample for negative
    int _up_sample;  // up sample for positive
    bool _is_update;  // if update param in minibatch, true is better, but false is good in theory
};

}
#endif
