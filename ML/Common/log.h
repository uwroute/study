/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-02-14 15:54
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
#include <stdint.h>

enum LOG_LEVEL
{
	TRACE = 0,
	DEBUG = 1,
	INFO = 2,
	WARNING = 3,
	ERROR = 4,
};

extern uint32_t log_level;

#define LOG_ERROR(fmt, ...) \
	if (log_level <= ERROR) fprintf(stdout, "[ERROR] " fmt"\n", __VA_ARGS__)

#define LOG_WARNING(fmt, ...) \
	if (log_level <= WARNING) fprintf(stdout, "[WARNING] " fmt"\n", __VA_ARGS__)

#define LOG_INFO(fmt, ...) \
	if (log_level <= INFO) fprintf(stdout, "[INFO] " fmt"\n", __VA_ARGS__)

#define LOG_DEBUG(fmt, ...) \
	if (log_level <= DEBUG) fprintf(stdout, "[DEBUG] " fmt"\n", __VA_ARGS__)

#define LOG_TRACE(fmt, ...) \
	if (log_level <= TRACE) fprintf(stdout, "[TRACE] " fmt"\n", __VA_ARGS__)

#endif  // _LOG_H_
