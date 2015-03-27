/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-21 10:08
#
# Filename: data.cpp
#
# Description: 
#
=============================================================================*/

#include <fstream>
#include <string>
#include "log.h"
#include "string_util.h"
#include "data.h"

#define CHECK_AND_DELETE(p) \
	if (p)					\
	{						\
		delete p;			\
		p = NULL;			\
	}

DataSet::~DataSet()
{
	CHECK_AND_DELETE(samples);
	CHECK_AND_DELETE(index);
	CHECK_AND_DELETE(label);
	label_size = 0;
	sample_size = 0;
}

int DataSet::load(const char* data_file)
{
	LOG_DEBUG("Start compute fea_size and sample_size in %s", data_file);
	std::ifstream infile(data_file);
	std::string line;
	getline(infile, line);
	while (!infile.eof())
	{
		std::vector<std::string> tmp;
		splitString(line, tmp, ' ');
		sample_size += tmp.size();
		sample_size++;
		label_size++;
		getline(infile, line);
	}
	infile.close();
	LOG_INFO("[sample_size=%u] and [label_size=%u]", sample_size, label_size);
	if (sample_size == label_size)
	{
		LOG_ERROR("[sample_size=%u] = [label_size=%u]", sample_size, label_size);
		return -1;
	}
	samples = new Feature[sample_size];
	if (!sample)
	{
		LOG_ERROR("Can't new %d bytes for sample!", sample_size*sizeof(Feature));
		return -1;
	}
	index = new uint32_t[label_size];
	if (!index)
	{
		LOG_ERROR("Can't new %d bytes for index!", label_size*sizeof(uint32_t));
		return -1;
	}
	label = new uint32_t[label_size];
	if (!label)
	{
		LOG_ERROR("Can't new %d bytes for label!", label_size*sizeof(uint32_t));
		return -1;
	}
	std::ifstream infile(data_file);
	getline(infile, line);
	uint32_t label_i = 0;
	uint32_t sample_i = 0;
	while (!infile.eof())
	{
		index[label_i] = sample_i;
		std::vector<std::string> vec;
		splitString(line, vec, ' ');
		label[label_i] = atof(vec[0]);
		Feature* fea = sample + sample_i;
		for (size_t i=1; i<vec.size(); ++i)
    	{
	        if (!sscanf(vec[i].c_str(), "%lu:%lf", &(fea->index), &(fea->value)))
	        {
	            continue;
	        }
	        fea_num = std::max(fea_num, fea->index);
	        ++fea;
	        ++sample_i;
    	}
    	getline(infile, line);
	}
	LOG_INFO("load %u feas and %u samples", sample_size, label_size);
}