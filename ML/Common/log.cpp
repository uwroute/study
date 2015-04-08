/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Wed 08 Apr 2015 03:44:27 PM CST [10.146.36.174]
#
# Filename: log.h
#
# Description: 
#
=============================================================================*/
#include "log.h"

std::string cur_time()
{
    time_t t = time(NULL);
    char buff[100];
    strftime(buff, 100, "%Y-%m-%d %H:%M:%S", localtime(&t));
    return std::string(buff);
}
