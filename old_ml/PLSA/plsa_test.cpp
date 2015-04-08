/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-07 10:58
#
# Filename: plsa_test.cpp
#
# Description:  test plsa
#
=============================================================================*/

#include <iostream>
#include <string>
#include "plsa.h"
#include <ctime>
#include <gflags/gflags.h>

using namespace ML;

DEFINE_string(data_file, "NULL", "train data");
DEFINE_string(model_file, "plsa.model", "model file");
DEFINE_int32(max_iter, 10, "max iter");
DEFINE_int32(topic_num, 10, "topic num");
DEFINE_int32(log_level, 0, "LogLevel :\n"
    "0 : TRACE"
    "1 : DEBUG"
    "2 : INFO"
    "3 : ERROR");

uint32_t log_level = 0;

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    PLSA model;
    model.init(FLAGS_topic_num, FLAGS_max_iter);
    model.load_data(FLAGS_data_file);
    model.train();
    model.save_model(FLAGS_model_file);
}
