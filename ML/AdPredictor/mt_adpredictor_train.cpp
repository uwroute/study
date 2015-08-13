#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "mt_adpredictor.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "adpredictor.model", "model file");
DEFINE_string(warm_model_file, "", "warm_model file");
DEFINE_double(init_mean, 0.0, "init_mean");
DEFINE_double(init_variance, 1.0, "init_variance");
DEFINE_double(beta, 1.0, "beta");
DEFINE_double(eps, 0.0, "eps");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(thread_num, 0, "thread_num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_int32(mini_batch, 0, "mini batch");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_bool(use_bias,  true, "if use bias");
DEFINE_double(bias, 1.0, "bias value");
DEFINE_int32(line_step, 100000, "log line step");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

void train_single(const std::string& file, ParallelAdPredictor& model)
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
    model.init(FLAGS_init_mean, FLAGS_init_variance,  FLAGS_beta, FLAGS_eps, FLAGS_mini_batch, FLAGS_max_fea_num, FLAGS_use_bias, FLAGS_bias);
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

void train(const std::string& file, ParallelAdPredictor& model)
{
    srand( (unsigned)time( NULL ) );
    LongDataSet data;
    int ret = load_data(file, data, FLAGS_sample_rate);
    if (ret != 0)
    {
        LOG_ERROR("Load Data File  %s Failed!", file.c_str());
        return;
    }
    model.init(FLAGS_init_mean, FLAGS_init_variance,  FLAGS_beta, FLAGS_eps, FLAGS_mini_batch, FLAGS_max_fea_num, FLAGS_use_bias, FLAGS_bias);
    time_t start = time(NULL);
    for (int i=0; i<data.sample_num; ++i) {
        LOG_DEBUG("Train  %d sample!", i);
        model.train(&(data.samples[data.sample_idx[i]]), data.labels[i]);
    }
    model.train(NULL, 0.0);
    model.join();
    time_t t = time(NULL)-start;
    LOG_INFO("Train cost time : %lu s", t);
    model.merge_model();
}

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    ML::ParallelAdPredictor model(FLAGS_thread_num);
    int32_t iter = 0;
    while (iter++ < FLAGS_max_iter)
    {
        if (FLAGS_thread_num == 0)
        {
            train_single(FLAGS_train_file, model);
        }
        else
        {
            train(FLAGS_train_file, model);
        }
    }
    model.save_model(FLAGS_model_file);
    return 0;
}