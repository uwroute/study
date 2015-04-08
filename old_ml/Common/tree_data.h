/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-21 10:08
#
# Filename: tree_data.h
#
# Description: 
#
=============================================================================*/
#ifndef _TREE_DATA_H_
#define _TREE_DATA_H_

enum FeaType
{
    MUTICLASS = 0,
    CONTINUES = 1,
};

struct TreeFeature
{
	int index;
	FeaType fea_type;
	double value;
};

struct TreeDataSet
{
	TreeFeature* samples; // samples : {1,0.5},{2,1.0},{-1,0.0},{2,1.0}
	int* index;  // samples index : 0, 3
	int* label;  // samples label : 1, 0
	int* feas;
};

#endif  // _TREE_DATA_H_
