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

enum RequestType {
    QUERY_PARAM = 1,
    UPDATE_PARAM = 2,
    END_TRAIN = 3,
    UNKNOW = 4,
};

struct Request {
public:
    RequestType type;
    int slave_seri;
    std::vector<uint64_t> messages_idx;
    std::vector<Message> messages_val;
public:
    void clear() {type=UNKNOW; slave_seri=-1; messages_idx.clear(); messages_val.clear();}
};

enum ResponseType {
    PUSH_PARAM = 1,
    TRAIN_SAMPLE = 2,
    TRAIN_FILE =3,
};

struct Response {
    ResponseType type;
    std::vector<uint64_t> messages_idx;
    std::vector<Message> messages_val;
    std::vector<LongFeature> sample;
    std::string train_file;
};

class AdPredictorMaster {
public:
    typedef std::unordered_map<uint64_t, Message> MessageHashMap;
    typedef std::unordered_map<uint64_t, Param> ParamHashMap;
public:
    AdPredictorMaster() :_bias(1.0), _rw_mutex(_m_mutex) {}
    ~AdPredictorMaster() {}
    void init(double mean, double variance, double beta, double eps, size_t max_fea_num = 1000*10000, bool ues_bias=true);
    void run();
    void SetRcvQueue(Common::MessageQueue<Request>* queue);
    void AddSlave(Common::MessageQueue<Response>* slave);
    void save_model(const std::string& file);
    int load_model(const std::string& file);
    // communication with slave
    void update_message(const Request& req);
    void get_message(const Request& req, Response& res);
private:
    void update_message(const uint64_t idx, const Message& param);
    void update_bias_message(const Message& param);
    Message get_message(uint64_t idx);
private:
    MessageHashMap _messages; 
    Message _prior_message;
    Message _bias_message;
    bool _use_bias;
    double _beta;
    double _eps;
    double _bias;
    Common::MessageQueue<Request>* _rcv_queue;
    std::vector<Common::MessageQueue<Response>*> _slave_queues;
    std::vector<bool> _slave_status;
private:
    pthread_rwlock_t _m_mutex;
    Common::RWMutex _rw_mutex;
};

class AdPredictorSlave {
public:
    typedef std::unordered_map<uint64_t, Message> MessageHashMap;
    typedef std::unordered_map<uint64_t, Param> ParamHashMap;
public:
    AdPredictorSlave() : _bias(1.0) {}
    ~AdPredictorSlave() {}
    void init(double mean, double variance, double beta, double eps, int mini_batch, size_t max_fea_num, bool ues_bias, double down_sample, bool update);
    void set_master(AdPredictorMaster* p) {_p_master = p;}
    void set_rcv_queue(Common::MessageQueue<Request>* rcv_queue) {_rcv_queue = rcv_queue;}
    void set_master_queue(Common::MessageQueue<Response>* master_queue) {_master_queue = master_queue;}
    void set_seri(int seri) {_seri = seri;}
    void train(LongFeature* sample, double label);
    void train(std::string& file);
    void train_minibatch(LongDataSet& data);
private:
    void update_message(const uint64_t idx, const Message& param);
    Message get_message(uint64_t idx);
    void set_param(const uint64_t idx, const Param& param);
    Param get_param(uint64_t idx);
    void active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance);
    double cumulative_probability(double  t, double mean=0.0, double variance=1.0);
    double gauss_probability(double t, double mean=0.0, double variance=1.0);
    // communication with master
    void form_query_request(LongDataSet& data, Request& req);
    void form_update_request(Request& req);
    void update_param(Response& res);
private:
    MessageHashMap _messages;
    Message _bias_message;
    ParamHashMap _params;
    Param _bias_param;
    Param _prior_param;
    double _beta;
    double _eps;
    double _bias;
    bool _use_bias;
    Common::MessageQueue<Request>* _rcv_queue;
    Common::MessageQueue<Response>* _master_queue;
    AdPredictorMaster* _p_master;
private:
    int _train_count;
    int _mini_batch;
    double _down_sample;
    bool _update;
    int _seri;
};

}
#endif