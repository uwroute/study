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

void AdPredictor::train(const LongFeature* sample, double label)
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
    while (sample->index !=  (uint64_t)-1)
    {
        CHECK_MAP(_w_mean, sample->index, _init_mean);
        CHECK_MAP(_w_variance, sample->index, _init_variance);

        double mean = _w_mean[sample->index];
        double variance = _w_variance[sample->index];

        mean += label*sample->value*variance/sqrt(total_variance)*v;
        variance *=  1 - sample->value*sample->value*variance/total_variance*w;

        double rectify_variance = _init_variance*variance/( (1-_eps)*_init_variance + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_init_mean/_init_variance );

        _w_mean[sample->index] = rectify_mean;
        _w_variance[sample->index] = rectify_variance;

        LOG_TRACE("fea_index : %lu,  mean : %lf, variance : %lf", sample->index, rectify_mean, rectify_variance);
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

double AdPredictor::predict(const LongFeature* sample, bool useEE)
{
    double total_mean=0.0, total_variance=0.0;
    if (useEE) {
        active_mean_variance_withEE(sample, total_mean, total_variance);
    }
    else {
        active_mean_variance(sample, total_mean, total_variance);
    }
    return cumulative_probability(total_mean/sqrt(total_variance), 0.0, 1.0);
}


void AdPredictor::save_model(const std::string& file)
{
    std::ofstream out_file(file.c_str());
    out_file << _init_mean << std::endl;
    out_file << _init_variance<< std::endl;
    out_file << _beta << std::endl;
    out_file << _eps << std::endl;
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

int AdPredictor::load_model(const std::string& file)
{
    std::ifstream infile(file.c_str());
    std::string line;
    // param
    getline(infile, line);
    _init_mean = atof(line.c_str());
    getline(infile, line);
    _init_variance = atof(line.c_str());
    getline(infile, line);
    _beta = atof(line.c_str());
    getline(infile, line);
    _eps = atof(line.c_str());
    getline(infile, line); 
    _USE_BIAS = atoi(line.c_str());
    getline(infile, line);  
    if (3 != sscanf(line.c_str(), "%lf\t%lf\t%lf", &_bias, &_bias_mean, &_bias_variance))
    {
        LOG_ERROR("Parser Bias Error :  %s", line.c_str());
        infile.close();
        return -1;
    }
    getline(infile, line); // get fea size
    int w_size = atoi(line.c_str());
    _w_mean.reserve(w_size);
    _w_variance.reserve(w_size);
    getline(infile, line);
    while (!infile.eof())
    {
        uint64_t fea = 0;
        double mean=0.0, variance=0.0;
        if (3 != sscanf(line.c_str(), "%lu\t%lf\t%lf", &fea, &mean, &variance) )
        {
            LOG_ERROR("Parser Weight Error :  %s", line.c_str());
            infile.close();
            return -1;
        }
        _w_mean[fea] = mean;
        _w_variance[fea] = variance;
        getline(infile, line);
    }
    infile.close();
    LOG_INFO("Load Mode Size : %lu", _w_mean.size());
    return 0;
}

void AdPredictor::active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance)
{
    total_mean = 0.0;
    total_variance = 0.0;
    while (sample->index != (uint64_t)-1)
    {
        if (_w_mean.end() != _w_mean.find(sample->index))
        {
            total_mean += _w_mean[sample->index] * sample->value;
            total_variance += _w_variance[sample->index] * sample->value*sample->value;
        }
        else
        {
            total_mean += _init_mean*sample->value;
            total_variance += _init_variance*sample->value*sample->value;
        }
        sample++;
    }
    if (_USE_BIAS)
    {
        total_mean += _bias_mean * _bias;
        total_variance += _bias_variance * _bias*_bias;
    }
    total_variance += _beta*_beta;
}

void AdPredictor::active_mean_variance_withEE(const LongFeature* sample, double& total_mean, double& total_variance)
{
    total_mean = 0.0;
    total_variance = 0.0;
    while (sample->index != (uint64_t)-1)
    {
        if (_w_mean.end() != _w_mean.find(sample->index))
        {
            total_mean += (sqrt(_w_variance[sample->index]) * gaussrand() + _w_mean[sample->index] )* sample->value;
        }
        else
        {
            total_mean += (sqrt(_init_variance) * gaussrand() + _init_mean )* sample->value;
        }
        sample++;
    }
    if (_USE_BIAS)
    {
        total_mean += (sqrt(_bias_variance) * gaussrand() + _bias_mean )* _bias;
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

void AdPredictor::merge(AdPredictor& other) {
    for (DoubleHashMap::const_iterator iter = other._w_mean.begin(); iter != other._w_mean.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        double other_m = iter->second;
        double other_v = other._w_variance[fea_idx];
        double cur_m = _init_mean;
        double cur_v = _init_variance;
        if (_w_mean.find(fea_idx) != _w_mean.end())
        {
            cur_m = _w_mean[fea_idx];
            cur_v = _w_variance[fea_idx];
        }
        double new_v = _init_variance/(_init_variance - other_v) * other_v;
        new_v = cur_v/(cur_v+new_v)*new_v;
        double new_w = new_v/cur_v*cur_m + new_v/other_v*other_m - new_v/_init_variance * _init_mean;
        _w_mean[fea_idx] = new_w;
        _w_variance[fea_idx] = new_v;
    }
    double other_m = other._bias_mean;
    double other_v = other._bias_variance;
    double cur_m = _bias_mean;
    double cur_v = _bias_variance;
    double new_v = _init_variance/(_init_variance - other_v) * other_v;
    new_v = cur_v/(cur_v+new_v)*new_v;
    double new_w = new_v/cur_v*cur_m + new_v/other_v*other_m - new_v/_init_variance * _init_mean;
    _bias_mean = new_w;
    _bias_variance = new_v;
}

void AdPredictor::copy(AdPredictor& other) {
    for (DoubleHashMap::const_iterator iter = other._w_mean.begin(); iter != other._w_mean.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        double other_m = iter->second;
        double other_v = other._w_variance[fea_idx];
        _w_mean[fea_idx] = other_m;
        _w_variance[fea_idx] = other_v;
    }
    _bias_mean = other._bias_mean;
    _bias_variance = other._bias_variance;
}

double AdPredictor::gaussrand()
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
        } while((S >= 1) || (fabs(S)  < 1e-10));
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
    {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return X;
}

void AdPredictor::update_message(AdPredictor& other)
{
    for (DoubleHashMap::const_iterator iter = other._w_mean_message.begin(); iter != other._w_mean_message.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        double message_m = iter->second;
        double message_v = other._w_variance_message[fea_idx];
        double cur_m = _init_mean;
        double cur_v = _init_variance;
        if (_w_mean.find(fea_idx) != _w_mean.end())
        {
            cur_m = _w_mean[fea_idx];
            cur_v = _w_variance[fea_idx];
        }
        double new_v = cur_v/(1.0 + message_v*cur_v);
        double new_w = new_v/cur_v*cur_m + new_v*message_m;
        _w_mean[fea_idx] = new_w;
        _w_variance[fea_idx] = new_v;
        LOG_TRACE("update_message : fea_index : %lu,  mean : %lf, variance : %lf", fea_idx, new_w, new_v);
    }
    double message_m = other._bias_mean_message;
    double message_v = other._bias_variance_message;
    double cur_m = _bias_mean;
    double cur_v = _bias_variance;
    double new_v = cur_v/(1.0 + message_v*cur_v);
    double new_w = new_v/cur_v*cur_m + new_v*message_m;
    _bias_mean = new_w;
    _bias_variance = new_v;
}
void AdPredictor::compute_message(const LongFeature* sample, double label)
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
    while (sample->index !=  (uint64_t)-1)
    {
        CHECK_MAP(_w_mean, sample->index, _init_mean);
        CHECK_MAP(_w_variance, sample->index, _init_variance);

        double mean = _w_mean[sample->index];
        double variance = _w_variance[sample->index];

        mean += label*sample->value*variance/sqrt(total_variance)*v;
        variance *=  1 - sample->value*sample->value*variance/total_variance*w;

        _w_variance_message[sample->index] += (_w_variance[sample->index] - variance)/(variance*_w_variance[sample->index]);
        _w_mean_message[sample->index] += mean/variance - _w_mean[sample->index]/_w_variance[sample->index];

        LOG_TRACE("fea_index : %lu,  mean : %lf, variance : %lf", sample->index, mean, variance);
        LOG_TRACE("fea_index : %lu,  mean_msg : %lf, variance_msg : %lf", sample->index, _w_mean_message[sample->index], _w_variance_message[sample->index]);
        sample++;
    }
    if (_USE_BIAS)
    {
        double mean = _bias_mean;
        double variance = _bias_variance;

        mean += label*_bias*variance/sqrt(total_variance)*v;
        variance *=  1 - fabs(_bias)*variance/total_variance*w;

        _bias_mean_message += mean/variance - _bias_mean/_bias_variance;
        _bias_variance_message += (_bias_variance - variance)/(variance*_bias_variance);
        
        LOG_TRACE("bias , mean_msg : %lf,  variance_msg : %lf", _bias_mean_message, _bias_variance_message);
    }
}
void AdPredictor::clear_message()
{
    _w_mean_message.clear();
    _w_variance_message.clear();
    _bias_mean_message = 0.0;
    _bias_variance_message = 0.0;
}

void AdPredictor::puring_model(int fea_num, double threshold)
{
    std::vector<uint64_t> filter_feature;
    filter_feature.reserve(_w_mean.size());
    for (DoubleHashMap::const_iterator iter = _w_mean.begin(); iter != _w_mean.end(); ++iter)
    {
        if (puring_feature(_w_mean[iter->first], _w_variance[iter->first], fea_num, threshold))
        {
            filter_feature.push_back(iter->first);
        }
    }
    for (size_t i=0; i<filter_feature.size(); ++i)
    {
        _w_mean.erase(filter_feature[i]);
        _w_variance.erase(filter_feature[i]);
    }
}

bool AdPredictor::puring_feature(double mean, double variance, int fea_num, double threshold)
{
    double prior_mean = _init_mean*fea_num;
    double prior_variance = _init_variance*fea_num + _beta*_beta;
    double poster_mean = _init_mean*(fea_num-1) + mean;
    double poster_variance = _init_variance*(fea_num-1) + variance + _beta*_beta;
    double prior_prob = cumulative_probability(prior_mean/sqrt(prior_variance));
    double poster_prob = cumulative_probability(poster_mean/sqrt(poster_variance));
    if (KL(prior_prob, poster_prob) < threshold)
    {
        return true;
    }
    return false;
}
double AdPredictor::KL(double p, double q)
{
    return p*(log(p) - log(q)) + (1-p)*(log(1-p) - log(1-q));
}

}
