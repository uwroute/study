/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 20:32
#
# Filename: matchbox_predict.cpp
#
# Description: 
#
=============================================================================*/
#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "matchbox.h"
#include "data/data.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(model_file, "matchbox.model", "model file");
DEFINE_string(test_file, "", "test file");
DEFINE_string(result_file, "", "test result file");
DEFINE_bool(dynamic,  false, "if dynamic predictor");
DEFINE_double(sample_rate, 1.0, "sample rate");
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

    MatchBox model;
    model.load_model(FLAGS_model_file);

    srand( (unsigned)time( NULL ) );

    std::vector<LongMatrixFeature> sample;
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
    LongMatrixFeature end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;
    while (!infile.eof())
    {
        sample.clear();
        label = 0.0;
        uint64_t ret = toSample(line, sample, label);
        if (ret > 0)
        {
            sample.push_back(end_fea);
            double pre_value = model.predict(&(sample[0]));
            if (FLAGS_dynamic)
            {
                if (label > 0.5 || (label < 0.5 && ( rand()*1.0/RAND_MAX < FLAGS_sample_rate) ) )
                {
                    model.train(&(sample[0]), label);
                }
            }
            ofile << label << " " << pre_value << endl;
        }
        getline(infile, line);
    }
    infile.close();
    ofile.close();
    return 0;
}
