/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:06
#
# Filename: data.cpp
#
# Description: 
#
=============================================================================*/

#include "data.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "Common/log.h"

namespace ML
{
using std::ifstream;

void splitString(const std::string& in, std::vector<std::string>& out, char s)
{
    size_t start = 0;
    size_t pos = in.find(s, start);
    while (pos != in.npos)
    {
        out.push_back(in.substr(start, pos-start));
        start = pos + 1;
        pos = in.find(s, start);
    }
    if (in.size() > start)
    {
        out.push_back(in.substr(start, in.size()-start));
    }
}

int toSample(const std::string& line, std::vector<Feature>& sample, double& label)
{
    std::vector<std::string> vec;
    splitString(line, vec, ' ');
    int MAX_FEA_NUM = 0;
    if (vec.size() <= 0 )
    {
        return 0;
    }
    label = atof(vec[0].c_str());
    Feature fea;
    for (size_t i=1; i<vec.size(); ++i)
    {
        fea.index = 0;
        fea.value = 0.0;
        if (sscanf(vec[i].c_str(), "%d:%lf", &(fea.index), &(fea.value)))
        {
            fea.index --;
            sample.push_back(fea);
        }
        MAX_FEA_NUM = std::max(MAX_FEA_NUM, fea.index);
    }
    if (sample.size() == 0)
    {
        return 0;
    }
    return MAX_FEA_NUM;
}

int load_data(const std::string& file, DataSet& data, bool is_compute_space)
{
    if (is_compute_space)
    {
    	calc_data_space(file, data);
    }	
    ifstream infile(file.c_str());
    data.samples.reserve(data.sample_fea_num);
    data.sample_idx.reserve(data.sample_num);
    data.labels.reserve(data.sample_num);
    if (!infile)
    {
        LOG_ERROR("Open data file : %s failed when compute space !", file.c_str());
        return -1;
    }
    // load data
    getline(infile, line);
    Feature end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;
    while (!infile.eof())
    {
        sample.clear();
        int ret = toSample(line, sample, label);
        if (ret > 0)
        {
            data.sample_idx.push_back(data.samples.size());
            data.samples.insert(data.samples.end(), sample.begin(), sample.end());
            data.samples.push_back(end_fea);
            data.labels.push_back(label);
            data.max_fea_num = std::max(data.max_fea_num, ret);
        }
        getline(infile, line);
    }
    data.max_fea_num ++;
    LOG_INFO("Load Data : Sample Fea Num = %lu", data.samples.size());
    LOG_INFO("Load Data : Sample Num = %lu", data.labels.size());
    LOG_INFO("Load Data : Max Fea Num = %d", data.max_fea_num);
    infile.close();
    return 0;
}
/*
 * calc data require space in file (bytes)
*/
int calc_data_space(const std::string& file, DataSet& data) 
{
    ifstream infile(file.c_str());
    if (!infile)
    {
        LOG_ERROR("Open data file : %s failed when compute space !", file.c_str());
        return -1;
    }
    std::string line;
    std::vector<Feature> sample;
    double label = 0.0;
    // compute data.sample_fea_num, data.sample_num
    getline(infile, line);
    while (!infile.eof())
    {
        sample.clear();
        int ret = toSample(line, sample, label);
        if (ret > 0)
        {
            data.sample_fea_num += sample.size();
            data.sample_fea_num += 1;
            data.sample_num += 1;
            data.max_fea_num = std::max(data.max_fea_num, ret);
        }
        getline(infile, line);
    }
    LOG_INFO("Compute Space : Sample Fea Num = %d", data.sample_fea_num);
    LOG_INFO("Compute Space : Sample Num = %d", data.sample_num);
    LOG_INFO("Compute Space : Max Fea Num = %d", data.max_fea_num);
    infile.close();
    return data.feature_fea_num*sizeof(Feature) + data.sample_num*(sizeof(int) + sizeof(double));
}

}