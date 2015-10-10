/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:07
#
# Filename: grad.h
#
# Description: 
#
=============================================================================*/
#ifndef _GRAD_H_
#define _GRAD_H_

#include <unordered_map>
#include "data.h"
#include "state.h"
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"

namespace ML
{

class GradCalcThread : public Common::Thread
{
public:
    // function for predict
    GradCalcThread() : _batch(1), _queue(NULL) {}
    ~GradCalcThread() {}
    virtual void run();
    void set_queue(SampleQueue* q) {_queue = q;}
    // function
    void calc_grad(const Feature* sample, double label);
    void calc_loss(const Feature* sample, double label);
    void calc_grad_and_loss(const Feature* sample, double label);
    void update_batch_grad();
    void update_batch_next_grad();
    void update_loss();
    void update_next_loss();
    void clear_state();
private:
    double wx(const Feature* sample);
    double log_loss(const Feature* sample, double label);
    double log_loss(const double wx, double label);
    double predict(const Feature* sample);
    double predict(const double wx);
    double get_w(int i);
    void process_batch();
private:
    std::unordered_map<int, double> _batch_grads;
    double _batch_loss;
    int _batch;
    int _calc_num;
    SampleQueue* _queue; 
};

}

#endif  // _GRAD_H_
