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

#include "data.h"
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"

namespace ML
{

class GradCalcThread : public Common::Thread
{
public:
    // function for predict
    GradCalcThread() : _batch(1) {}
    ~GradCalcThread() {}
    virtual void run();
    // function for opt
    void calc_grad(const Feature* sample, double label);
    void calc_loss(const Feature* sample, double label);
    void calc_grad_and_loss(const Feature* sample, double label);
    void update_batch_grad();
    void clear_state();
private:
    double wx(const Feature* sample);
    double log_loss(const Feature* sample, double label);
    double predict(const Feature* sample);
    double predict(const double wx);
    double get_w(int i);
    void update_grad(int i, double grad);
    void update_loss();
private:
    std::unordered_map<int, double> _batch_grads;
    double _batch_loss;
    int _batch;
    int _calc_num;
    SampleQueue* _queue; 
};

}

#endif  // _GRAD_H_
