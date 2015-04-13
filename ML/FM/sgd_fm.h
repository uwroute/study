/**
 * @author: liuxun
 * @last modified 2015-02-05 17:07
 * @file inc_fm.h
 * @description Recomendation Algorithm FM Model
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia. 
 */

#ifndef _SGD_FM_H_
#define _SGD_FM_H_

#include <iostream>
#include <cmath>
#include <map>
#include "data/data.h"

typedef std::map<uint64_t, double> DoubleHashMap;
typedef std::map<uint64_t, uint32_t> UintHashMap;
typedef std::map<uint64_t, std::vector<double> > DoubleHashMatrix;

namespace ML
{

class SgdFM
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
    uint32_t _num_factor;
    double _learn_rate;
    double _init_mean;
    double _init_stdev;
    std::vector<double> _sum;
    std::vector<double> _sum_square;
    // SGD
    UintHashMap _step;
    uint32_t _step0;
    double _decay_rate;
private:
    //SGD
    float cur_learn_rate(uint64_t step)
    {
        return _learn_rate/powf(1.0+sqrt(step*1.0), _decay_rate);
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
    int cov_string_to_v(std::string str, std::vector<double>& v);
public:
    SgdFM();
    ~SgdFM() {};
    int save_model(std::string filename);
    int load_model(std::string filename);
    int train(const Feature* sample, double label);
    double predict(const Feature* sample, bool is_train=false);
public:
    void set_reg(double regw, double reg0, double regv)
    {
        _regw = regw;
        _regv = regv;
        _reg0 = reg0;
    }
    void set_num_factor(uint32_t num_factor) {_num_factor=num_factor;}
    void set_decay_rate(double decay_rate) {_decay_rate = decay_rate;}
    void set_learn_rate(double lr) {_learn_rate=lr;}
};

}
#endif /* _SGD_FM_H_ */
