/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Wed 08 Apr 2015 05:09:43 PM CST [10.146.36.174]
#
# Filename: logistic.cpp
#
# Description: 
#
=============================================================================*/
#include <iostream>
#include <fstream>
#include "logistic_model.h"
#include "Common/log.h"
#include "data/data.h"
#include "gflags/gflags.h"

DEFINE_string(test_file, "NULL", "test data");
DEFINE_string(result_file, "NULL", "result data");
DEFINE_string(model_file, "lr.model", "model file");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;
using namespace std;

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;

    LogisticModel lr_model;
    lr_model.load_model(FLAGS_model_file.c_str());

    std::vector<Feature> sample;
    ifstream infile(FLAGS_test_file.c_str());
    if (!infile)
    {
        LOG_ERROR("Open test file : %s failed!", FLAGS_test_file.c_str());
        return -1;
    }
    ofstream ofile(FLAGS_result_file.c_str());
    if (!ofile)
    {
        LOG_ERROR("Open result file : %s failed!", FLAGS_result_file.c_str());
        return -1;
    }

    std::string line;
    getline(infile, line);
    double label = 0.0;
    Feature end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;
    while (!infile.eof())
    {
        sample.clear();
        label = 0.0;
        int ret = toSample(line, sample, label);
        if (ret > 0)
        {
            sample.push_back(end_fea);
            double pre_value = lr_model.predict(&(sample[0]));
            ofile << label << " " << pre_value << endl;
        }
        getline(infile, line);
    }
    infile.close();
    ofile.close();
    return 0;
}
