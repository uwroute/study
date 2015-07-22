#include "gflags/gflags.h"
#include <string>
#include "SparseBatchLR.h"
#include "Common/log.h"
#include <ctime>

// LR train params
DEFINE_string(train_file, "NULL", "train data");
DEFINE_string(model_file, "lr.model", "model file");
DEFINE_double(reg1, 0.0, "L1");
DEFINE_double(reg2, 0.0, "L2");
DEFINE_double(error, 1.0e-5, "error");
DEFINE_int32(max_iter, 100, "max iter");
DEFINE_int32(m, 0, "M");
DEFINE_int32(bias, 0.0, "bias");
DEFINE_int32(train_method, 0, "opt method :\n"
    "0 : GD"
    "1 : BFGS"
    "2 : OWLQN");
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

uint32_t log_level = 0;

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    SparseBatchLR model;
    model.set_bias(FLAGS_bias);
    model.set_reg1(FLAGS_reg1);
    model.set_reg2(FLAGS_reg2);
    model.set_max_iter(FLAGS_max_iter);
    model.set_error(FLAGS_error);
    model.set_train_method(FLAGS_train_method);
    model.set_linear_search_method(FLAGS_linear_search);
    if (FLAGS_train_method > 1)
    {
        if (FLAGS_m <= 0)
        {
            LOG_ERROR("LBFGS param error : [M = %d]", FLAGS_m);
            return -1;
        }
        model.set_M(FLAGS_m);
    }
    model.load_data_file(FLAGS_train_file);
    time_t start = time(NULL);
    model.train();
    time_t end = time(NULL);
    LOG_DEBUG("Train cost : %lu", end-start);
    model.save_model(FLAGS_model_file);
    return 0;
}
