#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "adpredictor.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "adpredictor.model", "model file");
DEFINE_double(init_mean, 0.0, "init_mean");
DEFINE_double(init_variance, 1.0, "init_variance");
DEFINE_double(beta, 1.0, "beta");
DEFINE_double(eps, 0.0, "eps");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

void train(const std::string& file, AdPredictor& model)
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
    std::vector<Feature> sample;
    double label = 0.0;
    Feature end_fea;
    end_fea.index = -1;
    end_fea.value = 0.0;

    getline(infile, line);
    while (!infile.eof())
    {
        sample.clear();
        label = 0.0;
        int ret = toSample(line, sample, label);
        if (ret > 0)
        {
            sample.push_back(end_fea);
            if (label < 0.5 && ( rand()*1.0/RAND_MAX > FLAGS_sample_rate) )
            {
                   getline(infile, line);
                   continue;
            }
            model.train(&(sample[0]), label);
        }
        getline(infile, line);
    }

    infile.close();
}

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    ML::AdPredictor model;
    model.init(FLAGS_init_mean, FLAGS_init_variance,  FLAGS_beta, FLAGS_eps, FLAGS_max_fea_num);
    int32_t iter = 0;
    while (iter++ < FLAGS_max_iter)
    {
        train(FLAGS_train_file, model);
    }
    model.save_model(FLAGS_model_file);
    return 0;
}