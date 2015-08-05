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
#ifndef _DATA_H_
#define _DATA_H_

#include <cstdio>
#include <stdint.h>
#include <vector>
#include <string>
#include <algorithm>

namespace ML
{
using std::vector;

struct Feature
{
    int index; // max fea num : 2147483648
    double value;
};

struct DataSet
{
    vector<Feature> samples; // one dim store type, end idx = -1 in a sample
    vector<size_t> sample_idx; // sample start idx
    vector<double> labels;   // one dim store type
    int sample_fea_num;      // size of samples
    int sample_num;          // sample num, size of labels
    int max_fea_num;         // max_fea for w size
    DataSet():sample_fea_num(0),sample_num(0),max_fea_num(0){}
};

struct LongFeature
{
    uint64_t index; // max fea num : 2^64
    double value;
};

struct LongDataSet
{
    vector<LongFeature> samples; // one dim store type, end idx = -1 in a sample
    vector<size_t> sample_idx; // sample start idx
    vector<double> labels;   // one dim store type
    int sample_fea_num;      // size of samples
    int sample_num;          // sample num, size of labels
    uint64_t max_fea_num;         // max_fea for w size
    LongDataSet():sample_fea_num(0),sample_num(0),max_fea_num(0){}
};

int toSample(const std::string& line, std::vector<Feature>& sample, double& label);
int load_data(const std::string& file, DataSet& data);
uint64_t toSample(const std::string& line, std::vector<LongFeature>& sample, double& label);
int load_data(const std::string& file, LongDataSet& data);

}

#endif  // _STRING_UTIL_H_
