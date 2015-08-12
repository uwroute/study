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
    for (int i=0; i<data.sample_num; ++i) {
        model.train(&(data.samples[data.sample_idx[i]]), data.labels[i]);
    }
    model.join();
    model.merge_model();
}

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    ML::ParallelAdPredictor model(FLAGS_thread_num);
    model.init(FLAGS_init_mean, FLAGS_init_variance,  FLAGS_beta, FLAGS_eps, FLAGS_max_fea_num, FLAGS_use_bias, FLAGS_bias);
    int32_t iter = 0;
    while (iter++ < FLAGS_max_iter)
    {
        train(FLAGS_train_file, model);
    }
    model.save_model(FLAGS_model_file);
    return 0;
}