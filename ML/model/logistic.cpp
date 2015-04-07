/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Tue 07 Apr 2015 05:34:02 PM CST [10.146.36.174]
#
# Filename: logistic.cpp
#
# Description: 
#
=============================================================================*/
#include <iostream>
#include "logistic_model.h"
#include "Common/log.h"
#include "data/data.h"
#include "opt/owlqn.h"
#include "gflags/gflags.h"

DEFINE_string(train_file, "NULL", "train data");
DEFINE_string(model_file, "lr.model", "model file");
DEFINE_double(reg1, 0.0, "L1");
DEFINE_double(reg2, 0.0, "L2");
DEFINE_double(error, 1.0e-5, "error");
DEFINE_int32(max_iter, 100, "max iter");
DEFINE_int32(m, 8, "M");
DEFINE_double(bias, 0.0, "bias");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;

    DataSet data;
    if (load_data(FLAGS_train_file, data))
    {
        LOG_ERROR("%s", "Load data file failed!");
        return -1;
    }

    LogisticModel lr_model;
    lr_model.set_l2(FLAGS_reg2);
    lr_model.set_dim(data.max_fea_num);
    
    OWLQN opt;
    opt.set_data(&data);
    opt.set_model(&lr_model);
    opt.set_l1(FLAGS_reg1);
    opt.set_max_iter(FLAGS_max_iter);
    opt.set_error(FLAGS_error);
    opt.set_m(FLAGS_m);
    opt.set_dim(data.max_fea_num);
    LOG_INFO("Opt require space : %d bytes!", opt.caluc_space());
    opt.init();
    opt.optimize();
    
    LOG_INFO("%s", "Save model!");
    lr_model.save_model(FLAGS_model_file.c_str());
    LOG_INFO("%s", "Save model end!");

    return 0;
}
