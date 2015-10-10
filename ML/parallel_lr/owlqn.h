/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:06
#
# Filename: owlqn.h
#
# Description: 
#
=============================================================================*/
#ifndef _OWLQN_H_
#define _OWLQN_H_

#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <iostream>
#include "Common/thread.h"
#include "Common/lock.h"
#include "state.h"

namespace ML
{

using std::vector;

// 0 idx is bias
struct ParamSet {
public:
    size_t N;
    vector<double> w;
    vector<double> next_w;
    vector<double> grad;
    vector<double> next_grad;
    double loss;
    double next_loss;
public:
    ParamSet() : _rw_mutex(_m_rw_mutex) {}
    // get
    double get_w(int i) {return w[i];}
    double get_next_w(int i) {return next_w[i];}
    double get_bias_w() {return w[0];}
    double get_next_bias_w() {return next_w[0];}
    // update
    void update_grad(int i, double g) {grad[i] += g;}
    void update_batch_grad(std::unordered_map<int, double> batch_grads);
    void update_next_grad(int i, double g) {next_grad[i] += g;}
    void update_batch_next_grad(std::unordered_map<int, double> batch_grads);
    void update_loss(double l) {loss+=l;}
    void update_next_loss(double l) {next_loss+=l;}
    // clear
    void clear_w() {}
    void clear_next_w() {}
    void clear_grad() {}
    void clear_next_grad() {}
    void clear_loss() {loss=0.0;}
    void clear_next_loss() {next_loss=0.0;}
private:
    pthread_rwlock_t _m_rw_mutex;
    Common::RWMutex _rw_mutex;
};

class OWLQN : public Common::Thread
{
    public:
        OWLQN(ParamSet& param): _N(param.N), _w(param.w), _next_w(param.next_w), _grad(param.grad), _next_grad(param.next_grad), 
            _steepest_dir(_next_grad), _loss(param.loss), _next_loss(param.next_loss) {}
        ~OWLQN(){}
    public:
        void optimize();
        void set_l1(double l1) {_l1=l1;}
        void set_l2(double l2) {_l2=l2;}
        void set_max_iter(int iter) {_max_iter=iter;}
        void set_m(int m) {_M = m;}
        void set_error(double e) {_error=e;}
        void set_dim(int dim) {_N = dim;}
        int init();
        int caluc_space();
        void save_model(std::string& model_file);
    public:
        void run();
    private:
        void makeSteepestDescDir(); // -grad dir
        void mapDirByInverseHessian(); // -Hk+1*grad
        void fixDirSign();  // -Hk+1*grad should be same dir with -grad
        void updateDir();	// compute _dir
        double checkDir();  // _dir should be desc dir
        void linearSearch();
        void shiftState();
        void getNextPoint(double alpha);
        void calc(GradThreadState grad_state);
        void l2grad(const vector<double>& w, vector<double>& grad);
        double l1Loss(const vector<double>& w, const double loss);
        double l2Loss(const vector<double>& w, const double loss);
        bool checkEnd();
    private:
        double dotProduct(vector<double>& x, vector<double>& y);
        void add(vector<double>& out, const vector<double>& in);
        void addScale(vector<double>& out, const vector<double>& x, const double scale);
        void addScaleInto(vector<double>& out, const vector<double>& x, const vector<double>& y, const double scale);
        void scale(vector<double>& out, const double scale);
        void scaleInto(vector<double>& out, const vector<double>& x, const double scale);
    private:
        // current param
        size_t& _N;
        vector<double>& _w;
        vector<double>& _next_w;
        // current dir
        vector<double>& _grad;
        vector<double>& _next_grad;
        vector<double>& _steepest_dir; // shared with next_grad, because of then can't exist in same times
        vector<double> _dir;
        double& _loss;
        double& _next_loss;
        // S,Y in LBFGS
        vector<double> _alpha;
        double _beta;
        vector<double> _sy;
        int _M;
        vector<vector<double> > _Y;
        vector<vector<double> > _S;
        int _start;
        int _end;
        // overfit
        double _l1;
        double _l2;
        // train param
        int _max_iter;
        int _cur_iter;
        double _error;
};
}
#endif  // _OWLQN_H_
