/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-02-13 00:01
#
# Filename: string_util.h
#
# Description: 
#
=============================================================================*/
#ifndef _STRING_UTIL_H_
#define _STRING_UTIL_H_

#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include "type.h"

namespace Common
{

struct Feature
{
    uint64_t index;
    double value;
};

enum FeaType
{
    MUTICLASS = 0,
    CONTINUES = 1,
};

struct TreeFeature
{
	uint64_t index;
	FeaType fea_type;
	double value;
};

struct DataSet
{
	Feature* samples; // samples : {1,0.5},{2,1.0},{-1,0.0},{2,1.0}
	uint64_t* index;  // samples index : 0, 3
	uint32_t* label;  // samples label : 1, 0
};

struct TreeDataSet
{
	TreeFeature* samples; // samples : {1,0.5},{2,1.0},{-1,0.0},{2,1.0}
	uint64_t* index;  // samples index : 0, 3
	uint32_t* label;  // samples label : 1, 0
};

void splitString(const std::string& in, std::vector<std::string>& out, char s);
uint64_t toSample(const std::string& line, std::map<uint64_t, double>& sample, double& label);
uint64_t toSample(const std::string& line, std::vector<Feature>& sample, double& label);
uint64_t toSample(const std::string& line, Feature* sample, double* label);

};

#endif  // _STRING_UTIL_H_
