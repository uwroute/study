/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Fri 03 Apr 2015 03:50:24 PM CST [10.146.36.174]
#
# Filename: data.h
#
# Description: 
#
=============================================================================*/
#ifndef _DATA_TPL_H_
#define _DATA_TPL_H_

#include <cstdio>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include "Common/string_util.h"

namespace ML
{
using std::vector;

// use template instead virtual fuction because of space
template <typename K, typename V>
struct Feature
{
public:
    typedef K IndexType;
    K index;
    V value;
public:
    int load(const char* str) {
        uint64_t idx = 0;
        double val = 0.0;
        if (2==sscanf(str, "%lu:%lf", &(idx), &(val)))
        {
            index = idx;
            value = val;
            return 0;
        }
        return -1;
    }
};

// use template instead virtual fuction because of space
template <typename K, typename V>
struct MFFeature
{
public:
    typedef K IndexType;
    K index;
    V value;
    int type;
public:
    int load(const char* str) {
        uint64_t idx = 0;
        double val = 0.0;
        if (3==sscanf(str, "%lu:%lf:%d", &(idx), &(val), &(type)))
        {
            index = idx;
            value = val;
            return 0;
        }
        return -1;
    }
};

template <typename FeatureType>
int toSample(const std::string& line, std::vector<FeatureType>& sample, double& label)
{
    std::vector<std::string> feature_vec;
    Common::splitString(line, feature_vec, ' ');
    if (feature_vec.size() <= 0 )
    {
        return -1;
    }
    label = atof(feature_vec[0].c_str());
    FeatureType feature;
    for (size_t i=1; i<feature_vec.size(); ++i)
    {
        if (-1 == feature.load(feature_vec[i].c_str()))
        {
            return -1;
        }
        sample.push_back(feature);
    }
    if (sample.size() == 0)
    {
        return -1;
    }
    return 0;
}

template<typename FeatureType>
struct DataSet
{
public:
    DataSet():sample_fea_num(0),sample_num(0),max_fea_num(0) {}
    int load(const std::string& file);
    int64_t calc_space(const std::string& file);
public:
    vector<FeatureType> samples; // one dim store type, end idx = -1 in a sample
    vector<size_t> sample_idx;       // sample start idx
    vector<double> labels;             // one dim store type
    int sample_fea_num;                // size of samples
    int sample_num;                      // sample num, size of labels
    typename FeatureType::IndexType max_fea_num;                    // max_fea for w size
};

template <typename FeatureType>
int64_t DataSet<FeatureType>::calc_space(const std::string& file) {
    int64_t space = 0;
    std::ifstream infile(file.c_str());
    if (!infile)
    {
        LOG_ERROR("Open data file : %s failed when compute space !", file.c_str());
        return -1;
    }
    std::string line;
    std::vector<FeatureType> sample;
    double label = 0.0;
    // compute data.sample_fea_num, data.sample_num
    getline(infile, line);
    while (!infile.eof())
    {
        sample.clear();
        int ret = toSample(line, sample, label);
        if (ret != -1)
        {
            this->sample_fea_num += sample.size();
            this->sample_fea_num += 1;
            this->sample_num += 1;
            for (size_t i=0; i<sample.size(); ++i)
            {
                this->max_fea_num = std::max(this->max_fea_num, sample[i].index);
            }
        }
        else
        {
            LOG_ERROR("Parse Sample Error : [ %s ] !", line.c_str());
        }
        getline(infile, line);
    }
    LOG_INFO("Compute Space : Sample Fea Num = %d", this->sample_fea_num);
    LOG_INFO("Compute Space : Sample Num = %d", this->sample_num);
    int64_t idx = max_fea_num;
    LOG_INFO("Compute Space : Max Fea Num = %ld", idx);
    infile.close();
    space = this->sample_fea_num * sizeof(FeatureType) + this->sample_num * (sizeof(size_t) + sizeof(double));
    LOG_INFO("Compute Space : Total Space  = %ld bytes", space);
    return space;
}

template <typename FeatureType>
int DataSet<FeatureType>::load(const std::string& file) {
    if (sample_fea_num == 0 && sample_num == 0 && max_fea_num == 0)
    {
        calc_space(file);
    }
    std::ifstream infile(file.c_str());
    if (!infile)
    {
        LOG_ERROR("Open data file : %s failed when compute space !", file.c_str());
        return -1;
    }
    std::string line;
    std::vector<FeatureType> sample;
    double label = 0.0;
    // load data
    this->samples.reserve(this->sample_fea_num);
    this->sample_idx.reserve(this->sample_num);
    this->labels.reserve(this->sample_num);
    // read data
    getline(infile, line);
    FeatureType end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;
    while (!infile.eof())
    {
        sample.clear();
        int ret = toSample(line, sample, label);
        if (ret != -1)
        {
            this->sample_idx.push_back(this->samples.size());
            this->samples.insert(this->samples.end(), sample.begin(), sample.end());
            this->samples.push_back(end_fea);
            this->labels.push_back(label);
            for (size_t i=0; i<sample.size(); ++i)
            {
                this->max_fea_num = std::max(this->max_fea_num, sample[i].index);
            }
        }
        else
        {
            LOG_ERROR("Parse Sample Error : [ %s ] !", line.c_str());
        }
        getline(infile, line);
    }
    LOG_INFO("Load Data : Sample Fea Num = %lu", this->samples.size());
    LOG_INFO("Load Data : Sample Num = %lu", this->labels.size());
    int64_t idx = max_fea_num;
    LOG_INFO("Compute Space : Max Fea Num = %ld", idx);
    infile.close();
    return 0;
}

}

#endif  // _DATA_TPL_H_
