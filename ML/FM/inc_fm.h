/**
 * @author: liuxun
 * @last modified 2015-02-05 17:07
 * @file inc_fm.h
 * @description Recomendation Algorithm FM Model
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia. 
 */

#ifndef _INC_FM_H_
#define _INC_FM_H_

#include <iostream>
#include <cmath>

typedef std::map<uint64_t, double> DoubleHashMap
typedef std::map<uint64_t, uint32_t> UintHashMap
typedef std::map<uint64_t, std::vector<double> > DoubleHashMatrix;

namespace ML
{
enum FM_TRAIN_METHOD {
    SGD = 0,
    FTRL = 1,
};

class IncFM
{
private:
    // model param
    double _w0;
    DoubleHashMap _w;
    DoubleHashMatrix _v;

    // model train param
    double _reg0;
    double _regv;
    double _regw;
    double _L1;
    uint32_t _num_factor;
    double _learn_rate;
    double _init_mean;
    double _init_stdev;
    std::vector<double> _sum;
    std::vector<double> _sum_square;
    int _train_method;
    // SGD
    UintHashMap _step;
    uint32_t step;
    double _decay_rate;

    // FTRL
    DoubleHashMap _z;
    DoubleHashMap _n;
    DoubleHashMatrix _vz;
    DoubleHashMatrix _vn;
    double _alpha;
    double _beta;

private:
    //SGD
    float cur_learn_rate(uint64_t step)
    {
        return _alpha/powf(_beta+sqrt(step*1.0), _decay_rate);
    }
    double gaussrand()
    {
        static double V1, V2, S;
        static int phase = 0;
        double X;
        if ( phase == 0 ) {
            do {
                double U1 = (double)rand() / RAND_MAX;
                double U2 = (double)rand() / RAND_MAX;
                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while(S >= 1 || S == 0);
            X = V1 * sqrt(-2 * log(S) / S);
        }
        else
        {
            X = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;
        return X;
    }
    void gauss_vec(std::vector<double>& vec)
    {
        for (size_t i=0; i<vec.size(); ++i)
        {
            vec[i] = gaussrand()*_init_stdev + _init_mean;
        }
    }
    double sgd_train(DoubleHashMap& ins, double label);
    double ftrl_train(DoubleHashMap& ins, double label);
    //bool is_zero(double x);
    //double L1(double w,double &u, double& q, double lr);
    //double penaltyL1(double w,double &u, double& q, double lr);
    //double lazyL1(double, double);
public:
    IncFM();
    ~IncFM() {};
    int save_model(std::string filename);
    int load_model(std::string filename);
    int train(DoubleHashMap& ins, double label);
    double predict(DoubleHashMap &ins, bool is_train=false);
public:
    void set_reg(double regw, double reg0, double regv)
    {
        _regw = regw;
        _regv = regv;
        _reg0 = reg0;
    }
    void set_L1(double l1) {_L1 = l1;}
    void set_num_factor(uint32_t num_factor) {_num_factor=num_factor;}
    void set_train_method(int method) {_train_method = method;}
    void set_decay_rate(double decay_rate) {_decay_rate = decay_rate;}
    void set_learn_rate(double alpha, double beta) 
    {
        _alpha = alpha;
        _beta = beta;
    }
};

}
#endif /* _INC_FM_H_ */
