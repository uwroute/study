//#include "ThirdParty/gflags/include/gflags/gflags.h"
#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "inc_fm.h"
#include "ML/Common/string_util.h"

// FTRL train params
DEFINE_string(method, "", "train data");
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "ftrl.model", "model file");
DEFINE_string(test_file, "", "test file");
DEFINE_string(test_result_file, "", "test result file");
DEFINE_double(L1, 0.03, "L1");
DEFINE_double(reg0, 1.0, "reg0");
DEFINE_double(regw, 1.0, "regw");
DEFINE_double(regv, 1.0, "regv");
DEFINE_double(alpha, 0.05, "lr alpha");
DEFINE_double(beta, 1.0, "lr beta");
DEFINE_int32(max_fea_num, 1000*1000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_int32(num_factor, 1, "num of factor");
DEFINE_double(decay_rate, 1.0, "decay rate");

void train(const std::string& file, FTRL& model);
void predict(const std::string& file, FTRL& model, const std::string& outfile);

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    ML::IncFM model;
    if (FLAGS_method == "train")
    {
        model.set_reg(FLAGS_regw, FLAGS_reg0, FLAGS_regv);
        model.set_num_factor(FLAGS_num_factor);
        model.set_train_method(FLAGS_train_method);
        model.set_decay_rate(FLAGS_decay_rate);
        model.set_learn_rate(FLAGS_alpha, FLAGS_beta);
        int32_t iter = 0;
        while (iter++ < FLAGS_max_iter)
        {
            train(FLAGS_train_file, model);
        }
        model.save_model(FLAGS_model_file);
    }
    else if (FLAGS_method == "predict")
    {
        model.load_model(FLAGS_model_file);
        predict(FLAGS_test_file, model, FLAGS_test_result_file);
    }
    else
    {
    }
    return 0;
}

void train(const std::string& file, FTRL& model)
{
    if (file.empty())
    {
        std::cout<< "data file is empty!" << std::endl;
        return;
    }
    std::ifstream infile(file.c_str());
    std::string line;
    // samples
    getline(infile, line);
    while (!infile.eof())
    {
        double label = 0.0;
        FTRL::DoubleHashMap sample;
        sample[0] = 1.0;
        if (!Common::toSample(line, sample, label))
        {
            if (label < 0)
            {
                label = 0.0;
            }
            model.train(sample, label);
        }
        getline(infile, line);
    }
}

void predict(const std::string& file, ML::IncFM& model, const std::string& outfile)
{
    if (file.empty())
    {
        std::cout<< "data file is empty!" << std::endl;
        return;
    }
    std::ifstream infile(file.c_str());
    std::ofstream fout(outfile.c_str());
    std::string line;

    getline(infile, line);
    while (!infile.eof())
    {
        double label = 0.0;
        FTRL::DoubleHashMap sample;
        sample[0] = 1.0;
        if (!Common::toSample(line, sample, label))
        {
            if (label < 0)
            {
                label = 0.0;
            }
            fout << label << " " << model.predict(sample) << std::endl;
        }
        getline(infile, line);
    }
}
