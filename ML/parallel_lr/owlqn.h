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
#include <iostream>

namespace ML
{

using std::vector;

struct ParamSet {
public:
    vector<double> w;
    vector<double> next_w;
    vector<double> grad;
    vector<double> next_grad;
    double loss;
    double next_loss;
public:
    
private:
    pthread_rwlock_t _m_rw_mutex;
    RWMutex _rw_mutex;
};

class OWLQN : public Common::Thread
{
    public:
        OWLQN():_steepest_dir(_next_grad){}
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
        double l1Loss(const vector<double>& w, const double loss);
        double l2loss(const vector<double>& w, const double loss);
        void getNextPoint(double alpha);
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
        size_t _N;
        vector<double> _w;
        vector<double> _next_w;
        // current dir
        vector<double> _dir;
        vector<double> _grad;
        vector<double> _next_grad;
        vector<double>& _steepest_dir; // shared with next_grad, because of then can't exist in same times
        double _loss;
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
