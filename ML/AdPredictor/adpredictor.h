/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-07-22 16:46
#
# Filename: adpredictor.h
#
# Description: adpredictor
#
=============================================================================*/

#ifndef __ADPREDICTOR_H__
#define __ADPREDICTOR_H__

#include <map>
// #include <tr1/unordered_map>
#include <string>
#include <stdint.h>
#include <unordered_map>
#include "data/data_tpl.hpp"

namespace ML
{
typedef Feature<uint64_t, double> LongFeature;
// using std::tr1::unordered_map
class AdPredictor {
public:
    typedef std::unordered_map<uint64_t, double> DoubleHashMap;
    //typedef std::map<uint64_t, double> DoubleHashMap;
public:
    AdPredictor() : _init_mean(0.0), _init_variance(1.0), _beta(1.0), _eps(0.0), _bias_mean(_init_mean), _bias_variance(_init_variance), _bias(1.0) {}
    ~AdPredictor(){}
    void init(double mean, double variance, double beta, double eps, size_t max_fea_num = 1000*10000, bool ues_bias=true, double bias=1.0);
    void train(const LongFeature* sample, double label);
    double predict(const LongFeature* sample, bool useEE = false);
    void save_model(const std::string& file);
    int load_model(const std::string& file);
    void puring_model(int fea_num, double threshold);
public:
    void set_init_mean(double mean) {_init_mean=mean;}
    void set_init_variance(double variance) {_init_variance=variance;}
    void set_beta(double beta) {_beta=beta;}
    void active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance);
    void active_mean_variance_withEE(const LongFeature* sample, double& total_mean, double& total_variance);
    double cumulative_probability(double  t, double mean=0.0, double variance=1.0);
    double gauss_probability(double t, double mean=0.0, double variance=1.0);
    double gaussrand();
    bool puring_feature(double mean, double variance, int fea_num, double threshold);
    double KL(double p, double q); // KL(p||q)
public:
    // for Parallel
    void merge(AdPredictor& other);
    void copy(AdPredictor& other);
    void update_message(AdPredictor& other);
    void compute_message(const LongFeature* sample, double label);
    void clear_message();
private:
    DoubleHashMap _w_mean;
    DoubleHashMap _w_variance;
    DoubleHashMap _w_mean_message;
    DoubleHashMap _w_variance_message;
    double _init_mean;
    double _init_variance;
    double _beta;
    double _eps;
    //bias para
    bool _USE_BIAS;
    double _bias_mean;
    double _bias_variance;
    double _bias_mean_message;
    double _bias_variance_message;
    double _bias;
};

}
#endif
