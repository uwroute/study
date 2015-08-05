#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "ftrl.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "ftrl.model", "model file");
DEFINE_double(lamda1, 0.03, "L1");
DEFINE_double(lamda2, 1.0, "L2");
DEFINE_double(alpha, 0.05, "lr alpha");
DEFINE_double(beta, 1.0, "lr beta");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_bool(use_bias,  true, "if use bias");
DEFINE_bool(reg_bias, false, "if reg bias");
DEFINE_double(bias, 1.0, "bias value");
DEFINE_int32(line_step, 50000, "log line step");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

void train(const std::string& file, FTRL& model)
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
    std::vector<LongFeature> sample;
    double label = 0.0;
    LongFeature end_fea;
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
    ML::FTRL model;
    model.init(FLAGS_alpha, FLAGS_beta, FLAGS_lamda1, FLAGS_lamda2, FLAGS_max_fea_num, FLAGS_use_bias, FLAGS_reg_bias, FLAGS_bias);
    int32_t iter = 0;
    while (iter++ < FLAGS_max_iter)
    {
        train(FLAGS_train_file, model);
    }
    model.save_model(FLAGS_model_file);
    return 0;
}

