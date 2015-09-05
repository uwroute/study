/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:06
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
#include <map>
#include <algorithm>
#include "Common/thread.h"
#include "Common/lock.h"
#include "Common/msg_queue.h"

using std::vector;

struct Feature
{
    int index; // max fea num : 2147483648
    double value; // fea val
};

struct DataSet
{
    vector<Feature> samples; // one dim store type, end idx = -1 in a sample
    vector<size_t> sample_idx; // sample start idx
    vector<double> labels;   // one dim store type
    uint64_t sample_fea_num;      // size of samples
    int sample_num;          // sample num, size of labels
    int max_fea_num;         // max_fea for w size
    DataSet():sample_fea_num(0),sample_num(0),max_fea_num(0){}
};

struct Sample
{
     Feature* x;
     double y;
};

typedef Common::MessageQueue<Sample> SampleQueue;

int toSample(const std::string& line, std::vector<Feature>& sample, double& label);
int load_data(const std::string& file, DataSet& data, bool is_compute_space = true);
int calc_data_space(const std::string& file, DataSet& data);

#endif  // _DATA_H_
