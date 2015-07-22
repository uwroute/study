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
#include "data/data.h"

namespace ML
{
// using std::tr1::unordered_map
class AdPredictor {
public:
    // typedef unordered_map<uint64_t, double> DoubleHashMap;
    typedef std::map<uint64_t, double> DoubleHashMap;
public:
    AdPredictor() {}
    ~AdPredictor(){}
    void init(double mean, double variance, double beta, double eps, size_t max_fea_num = 1000*10000);
    void train(const Feature* sample, double label);
    double predict(const Feature* sample);
    void save_model(const std::string& file);
    void load_model(const std::string& file);
public:
    void set_init_mean(double mean) {_init_mean=mean;}
    void set_init_variance(double variance) {_init_variance=variance;}
    void set_beta(double beta) {_beta=beta;}
    void active_mean_variance(const Feature* sample, double& total_mean, double& total_variance);
    double cumulative_probability(double  t, double mean, double variance);
    double gauss_probability(double t, double mean, double variance);
private:
    DoubleHashMap _w_mean;
    DoubleHashMap _w_variance;
    double _init_mean;
    double _init_variance;
    double _beta;
    double _eps;
};

}
#endif