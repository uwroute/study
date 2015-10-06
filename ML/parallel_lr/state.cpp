/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-10-05 15:04
#
# Filename: state.cpp
#
# Description: 
#
=============================================================================*/

#include "state.h"

void GradThreadStatus::add_done_num()
{
	Lock lock(_mutex);
	_done_num++;
}
void GradThreadStatus::init_done_num()
{
	Lock lock(_mutex);
	_done_num=0;
}