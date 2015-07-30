/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-07-22 16:53
#
# Filename: adpredictor.cpp
#
# Description: adpredictor
#
=============================================================================*/

#include "adpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"

namespace ML {

void AdPredictor::init(double mean, double variance, double beta, double eps, size_t max_fea_num, bool use_bias, double bias)
{
    _w_mean.reserve(max_fea_num);
    _w_variance.reserve(max_fea_num);
    _init_mean = mean;
    _init_variance = variance;
    _bias_mean = mean;
    _bias_variance = variance;
    _beta = beta;
    _eps  = eps;
    _USE_BIAS = use_bias;
    _bias = bias;
}

#define CHECK_MAP(map, key, value) \
    if (map.end() == map.find(key)) \
    {                               \
        map[key] = value;             \
    }

void AdPredictor::train(const Feature* sample, double label)
{
    if (label < 0.5)
    {
        label = -1.0;
    }
    double total_mean=0.0, total_variance=0.0;
    active_mean_variance(sample, total_mean, total_variance);
    LOG_TRACE("total_mean : %lf, total_variance : %lf", total_mean, total_variance);
    double t = label*total_mean/sqrt(total_variance);
    if (fabs(t) > 5.0)
    {
        t = t < 0 ? -5.0 : 5.0;
    }
    double v = gauss_probability(t, 0.0, 1.0) / cumulative_probability(t, 0.0, 1.0);
    double w = v*(v + t);
    LOG_TRACE("v : %lf, w : %lf", v, w);
    while (sample->index != -1)
    {
        CHECK_MAP(_w_mean, sample->index, _init_mean);
        CHECK_MAP(_w_variance, sample->index, _init_variance);

        double mean = _w_mean[sample->index];
        double variance = _w_variance[sample->index];

        mean += label*sample->value*variance/sqrt(total_variance)*v;
        variance *=  1 - fabs(sample->value)*variance/total_variance*w;

        double rectify_variance = _init_variance*variance/( (1-_eps)*_init_variance + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_init_mean/_init_variance );

        _w_mean[sample->index] = rectify_mean;
        _w_variance[sample->index] = rectify_variance;

        LOG_TRACE("fea_index : %d,  mean : %lf, variance : %lf", sample->index, rectify_mean, rectify_variance);
        sample++;
    }
    if (_USE_BIAS)
    {
        double mean = _bias_mean;
        double variance = _bias_variance;

        mean += label*_bias*variance/sqrt(total_variance)*v;
        variance *=  1 - fabs(_bias)*variance/total_variance*w;

        double rectify_variance = _init_variance*variance/( (1-_eps)*_init_variance + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_init_mean/_init_variance );

        _bias_mean = rectify_mean;
        _bias_variance = rectify_variance;

        LOG_TRACE("bias , mean : %lf,  variance : %lf", rectify_mean, rectify_variance);
    }
}

double AdPredictor::predict(const Feature* sample)
{
    double total_mean=0.0, total_variance=0.0;
    active_mean_variance(sample, total_mean, total_variance);
    return cumulative_probability(total_mean/sqrt(total_variance), 0.0, 1.0);
}

void AdPredictor::save_model(const std::string& file)
{
    std::ofstream out_file(file.c_str());
    out_file << _init_mean << std::endl;
    out_file << _init_variance<< std::endl;
    out_file << _USE_BIAS << std::endl;
    out_file << _bias << "\t"
            << _bias_mean << "\t"
            << _bias_variance << std::endl;
    out_file << _w_mean.size() << std::endl;
    for (DoubleHashMap::const_iterator iter = _w_mean.begin(); iter != _w_mean.end(); ++iter)
    {
        out_file << iter->first << "\t" 
            << _w_mean[iter->first] << "\t"
            << _w_variance[iter->first] << std::endl;
    }
}

void AdPredictor::load_model(const std::string& file)
{
    std::ifstream infile(file.c_str());
    std::string line;
    // param
    getline(infile, line);
    _init_mean = atof(line.c_str());
    getline(infile, line);
    _init_variance = atof(line.c_str());
    getline(infile, line);  // get fea size
    sscanf(line.c_str(), "%lf\t%lf\t%lf", &_bias, &_bias_mean, &_bias_variance);
    getline(infile, line);
    int w_size = atoi(line.c_str());
    _w_mean.reserve(w_size);
    _w_variance.reserve(w_size);
    while (!infile.eof())
    {
        uint64_t fea = 0;
        double mean=0.0, variance=0.0;
        sscanf(line.c_str(), "%lu\t%lf\t%lf", &fea, &mean, &variance);
        _w_mean[fea] = mean;
        _w_variance[fea] = variance;
        getline(infile, line);
    }
}

void AdPredictor::active_mean_variance(const Feature* sample, double& total_mean, double& total_variance)
{
    total_mean = 0.0;
    total_variance = 0.0;
    while (sample->index != -1)
    {
        if (_w_mean.end() != _w_mean.find(sample->index))
        {
            total_mean += _w_mean[sample->index] * sample->value;
            total_variance += _w_variance[sample->index] * fabs(sample->value);
        }
        else
        {
            total_mean += _init_mean;
            total_variance += _init_variance;
        }
        sample++;
    }
    if (_USE_BIAS)
    {
        total_mean += _bias_mean * _bias;
        total_variance += _bias_variance * fabs(_bias);
    }
    total_variance += _beta*_beta;
}

double AdPredictor::cumulative_probability(double  t, double mean, double variance)
{
    double m = (t - mean);
    if (fabs(m) > 40*sqrt(variance) )
    {
        return m < 0 ? 0.0 : 1.0;
    }
    return 0.5*(1 + erf( m / sqrt(2*variance) ) );
}

double AdPredictor::gauss_probability(double t, double mean, double variance)
{
    const double PI = 3.1415926;
    double m = (t - mean) / sqrt(variance);
    return exp(-m*m/2) / sqrt(2*PI*variance);
}

}
