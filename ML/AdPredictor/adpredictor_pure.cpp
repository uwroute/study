#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "adpredictor.h"
#include "data/data.h"
#include "Common/log.h"

// FTRL train params
DEFINE_string(model_file, "adpredictor.model", "model file");
DEFINE_string(pure_model, "adpredictor_pure.model", "pure model file");
DEFINE_int32(fea_num, 49, "non zero feature num");
DEFINE_double(threshold, 1.0e-4, "threshold");
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

    AdPredictor model;
    if (-1 == model.load_model(FLAGS_model_file))
    {
        LOG_ERROR("load model %s failed!", FLAGS_model_file.c_str());
        return -1;
    }
    LOG_INFO("load model %s successful!", FLAGS_model_file.c_str());

    srand( (unsigned)time( NULL ) );

    model.puring_model(FLAGS_fea_num, FLAGS_threshold);
    model.save_model(FLAGS_pure_model);
    return 0;
}