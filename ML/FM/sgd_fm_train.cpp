#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "sgd_fm.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "fm.model", "model file");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_int32(num_factor, 0, "num factor");
DEFINE_double(reg0, 0.0, "reg for bias");
DEFINE_double(regw, 0.0, "reg for w");
DEFINE_double(regv, 0.0, "reg for v");
DEFINE_double(decay_rate, 1.0, "decay_cate");
DEFINE_double(learn_rate, 0.05, "decay_cate");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

void train(const std::string& file, SgdFM& model)
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
    ML::SgdFM model;
    model.set_reg(FLAGS_regw, FLAGS_reg0, FLAGS_regv);
    model.set_num_factor(FLAGS_num_factor);
    model.set_decay_rate(FLAGS_decay_rate);
    model.set_learn_rate(FLAGS_learn_rate);
    int32_t iter = 0;
    while (iter++ < FLAGS_max_iter)
    {
        train(FLAGS_train_file, model);
    }
    model.save_model(FLAGS_model_file);
    return 0;
}

