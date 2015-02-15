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
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

namespace Common
{

void splitString(const std::string& in, std::vector<std::string>& out, char s);
uint64_t toSample(const std::string& line, std::map<uint64_t, double>& sample, double& label);

};

#endif  // _STRING_UTIL_H_
