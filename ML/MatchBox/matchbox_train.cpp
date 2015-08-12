/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 20:32
#
# Filename: matchbox_train.cpp
#
# Description: 
#
=============================================================================*/

#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "matchbox.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "matchbox.model", "model file");
DEFINE_string(warm_model_file, "", "warm_model file");
DEFINE_int32(k, 0, "k dim");
DEFINE_string(prior_mean, "0.0,0.0,0.0", "prior_mean");
DEFINE_string(prior_variance, "1.0,1.0,1.0", "prior_variance");
DEFINE_double(beta, 1.0, "beta");
DEFINE_double(eps, 0.0, "eps");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_int32(line_step, 100000, "log line step");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

void train(const std::string& file, MatchBox& model)
{
    if (file.empty())
    {
        std::cout<< "data file is empty!" << std::endl;
        return;
    }
    std::ifstream infile(file.c_str());
    if (!infile)
    {
        LOG_ERROR("Load data file : %s failed!", file.c_str());
        return;
    }
    srand( (unsigned)time( NULL ) );
    // samples
    std::string line;
    std::vector<LongMatrixFeature> sample;
    double label = 0.0;
    LongMatrixFeature end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;

    int line_count = 0;
    time_t start = time(NULL);
    getline(infile, line);
    while (!infile.eof())
    {
        sample.clear();
        label = 0.0;
        uint64_t ret = toSample(line, sample, label);
        if (ret > 0)
        {
            sample.push_back(end_fea);
            if (label < 0.5 && ( rand()*1.0/RAND_MAX > FLAGS_sample_rate) )
            {
                   getline(infile, line);
                   continue;
            }
            model.train(&(sample[0]), label);
            line_count ++;
            if (line_count%FLAGS_line_step == 0)
            {
                time_t end = time(NULL);
                LOG_INFO("Train Lines : %d cost %d ms", line_count, int(end-start));
            }
        }
        getline(infile, line);
    }

    infile.close();
}

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    ML::MatchBox model;
    model.init(FLAGS_k, FLAGS_prior_mean, FLAGS_prior_variance,  FLAGS_beta, FLAGS_max_fea_num);
    int32_t iter = 0;
    LOG_INFO("%s", "Train MatchBox model start!");
    while (iter++ < FLAGS_max_iter)
    {
        train(FLAGS_train_file, model);
    }
    LOG_INFO("%s", "Train MatchBox model end!");
    LOG_INFO("%s", "Save MatchBox model start!");
    model.save_model(FLAGS_model_file);
    LOG_INFO("%s", "Save MatchBox model end!");
    return 0;
}
