/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Wed 08 Apr 2015 03:46:06 PM CST [10.146.36.174]
#
# Filename: log.h
#
# Description: 
#
=============================================================================*/
#ifndef _LOG_H_
#define _LOG_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include "type.h"

enum LOG_LEVEL
{
	TRACE = 0,
	DEBUG = 1,
	INFO = 2,
	WARNING = 3,
	ERROR = 4,
};

extern uint32_t log_level;

std::string cur_time();

#define LOG_ERROR(fmt, ...) \
	if (log_level <= ERROR) fprintf(stdout, "[ERROR][%s] " fmt"\n", cur_time().c_str(), __VA_ARGS__)

#define LOG_WARNING(fmt, ...) \
	if (log_level <= WARNING) fprintf(stdout, "[WARNING][%s] " fmt"\n", cur_time().c_str(), __VA_ARGS__)

#define LOG_INFO(fmt, ...) \
	if (log_level <= INFO) fprintf(stdout, "[INFO][%s] " fmt"\n", cur_time().c_str(), __VA_ARGS__)

#define LOG_DEBUG(fmt, ...) \
	if (log_level <= DEBUG) fprintf(stdout, "[DEBUG][%s] " fmt"\n", cur_time().c_str(), __VA_ARGS__)

#define LOG_TRACE(fmt, ...) \
	if (log_level <= TRACE) fprintf(stdout, "[TRACE][%s] " fmt"\n", cur_time().c_str(), __VA_ARGS__)

#endif  // _LOG_H_
