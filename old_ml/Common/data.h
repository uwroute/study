/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-21 10:07
#
# Filename: data.h
#
# Description: 
#
=============================================================================*/
#ifndef _DATA_H_
#define _DATA_H_

struct Feature
{
    int32_t index;
    double value;
};

struct DataSet
{
	Feature* samples; // samples : {1,0.5},{2,1.0},{-1,0.0},{2,1.0}
	uint32_t* index;  // samples index : 0, 3
	uint32_t* label;  // samples label : 1, 0
	uint32_t sample_size; // len of samples
	uint32_t label_size; // len of index and label
	uint32_t fea_num;	// num of fea
	DataSet():samples(NULL), index(NULL), label(NULL), label_size(0), sample_size(0),fea_num(0) {}
	~DataSet();
	int loadData(const char* data_file);
	Feature* get_sample(uint32_t i) {return samples + index[i];}
};

#endif  // _DATA_H_
