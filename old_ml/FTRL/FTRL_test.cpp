//#include "ThirdParty/gflags/include/gflags/gflags.h"
#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include "FTRL.h"
#include "ML/Common/string_util.h"

// FTRL train params
DEFINE_string(method, "", "train data");
DEFINE_string(train_file, "", "train data");
DEFINE_string(model_file, "ftrl.model", "model file");
DEFINE_string(test_file, "", "test file");
DEFINE_string(test_result_file, "", "test result file");
DEFINE_double(lamda1, 0.03, "L1");
DEFINE_double(lamda2, 1.0, "L2");
DEFINE_double(alpha, 0.05, "lr alpha");
DEFINE_double(beta, 1.0, "lr beta");
DEFINE_int32(max_fea_num, 1000*1000, "fea num");
DEFINE_int32(max_iter, 1, "max iter");

void train(const std::string& file, FTRL& model);
void predict(const std::string& file, FTRL& model, const std::string& outfile);

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    FTRL model;
    if (FLAGS_method == "train")
    {
        model.init(FLAGS_alpha, FLAGS_beta, FLAGS_alpha, FLAGS_beta, FLAGS_max_fea_num);
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
        if (Common::toSample(line, sample, label))
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

void predict(const std::string& file, FTRL& model, const std::string& outfile)
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
        if (Common::toSample(line, sample, label))
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
