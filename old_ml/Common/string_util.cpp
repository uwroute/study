/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-02-13 01:05
#
# Filename: string_util.cpp
#
# Description: 
#
=============================================================================*/
#include "string_util.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace Common
{

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

uint64_t toSample(const std::string& line, std::map<uint64_t, double>& sample, double& label)
{
    std::vector<std::string> vec;
    splitString(line, vec, ' ');
    uint64_t MAX_FEA_NUM = 0;
    if (vec.size() <= 0 )
    {
        return 0;
    }
    label = atof(vec[0].c_str());
    for (size_t i=1; i<vec.size(); ++i)
    {
        uint64_t k = 0;
        double v = 0.0;
        if (sscanf(vec[i].c_str(), "%lu:%lf", &k, &v))
        {
            sample[k-1] = v;
        }
        MAX_FEA_NUM = std::max(MAX_FEA_NUM, k);
    }
    if (sample.size() == 0)
    {
        return 0;
    }
    return MAX_FEA_NUM;
}

uint64_t toSample(const std::string& line, std::vector<Feature>& sample, double& label)
{
    sample.clear();
    std::vector<std::string> vec;
    splitString(line, vec, ' ');
    uint64_t MAX_FEA_NUM = 0;
    if (vec.size() <= 0 )
    {
        return 0;
    }
    label = atof(vec[0].c_str());
    for (size_t i=1; i<vec.size(); ++i)
    {
        Feature fea;
        if (sscanf(vec[i].c_str(), "%lu:%lf", &(fea.index), &(fea.value)))
        {
            sample.push_back(fea);
        }
        else
        {
            continue;
        }
        MAX_FEA_NUM = std::max(MAX_FEA_NUM, fea.index);
    }
    if (sample.size() == 0)
    {
        return 0;
    }
    return MAX_FEA_NUM;
}

uint64_t toSample(const std::string& line, Feature* sample, double* label)
{
    std::vector<std::string> vec;
    splitString(line, vec, ' ');
    uint64_t MAX_FEA_NUM = 0;
    if (vec.size() <= 0 )
    {
        return 0;
    }
    *label = atof(vec[0].c_str());
    Feature* fea = sample;
    for (size_t i=1; i<vec.size(); ++i)
    {
        if (!sscanf(vec[i].c_str(), "%lu:%lf", &(fea->index), &(fea->value)))
        {
            continue;
        }
        MAX_FEA_NUM = std::max(MAX_FEA_NUM, fea->index);
        ++fea;
    }
    fea->index = -1;
    fea->value = 0.0;
    if ((fea-sample) == 0)
    {
        return 0;
    }
    return MAX_FEA_NUM;
}
};
