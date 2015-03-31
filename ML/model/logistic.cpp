/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-31 21:15
#
# Filename: logistic.cpp
#
# Description: 
#
=============================================================================*/
#include <iostream>
#include "logistic_model.h"
#include "data/data.h"
#include "opt/owlqn.h"
#include "gflags/gflags.h"

DEFINE_string(train_file, "NULL", "train data");
DEFINE_string(model_file, "lr.model", "model file");
DEFINE_double(reg1, 0.0, "L1");
DEFINE_double(reg2, 0.0, "L2");
DEFINE_double(error, 1.0e-5, "error");
DEFINE_int32(max_iter, 100, "max iter");
DEFINE_int32(m, 0, "M");
DEFINE_int32(bias, 0.0, "bias");
// LinearSearch params
DEFINE_int32(linear_search, 0, "linear search method : \n"
        "0 : Armijo linear search\n"
        "1 : simple linear search\n"
        "2 : Wolf linear search\n"
        "3 : Wolf strong linear search\n"
        );
DEFINE_int32(log_level, 0, "LogLevel :\n"
    "0 : TRACE"
    "1 : DEBUG"
    "2 : INFO"
    "3 : ERROR");

int main()
{

}
