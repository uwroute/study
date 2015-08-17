#include "gflags/gflags.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "parallel_adpredictor.h"
#include "Common/log.h"
#include "Common/string_util.h"

// FTRL train params
DEFINE_string(train_files, "", "train data");
DEFINE_string(model_file, "padpredictor.model", "model file");
DEFINE_string(warm_model_file, "", "warm_model file");
DEFINE_double(init_mean, 0.0, "init_mean");
DEFINE_double(init_variance, 1.0, "init_variance");
DEFINE_double(beta, 1.0, "beta");
DEFINE_double(eps, 0.0, "eps");
DEFINE_int32(max_fea_num, 1000*10000, "fea num");
DEFINE_int32(slave_num, 0, "thread_num");
DEFINE_int32(max_iter, 1, "max iter");
DEFINE_int32(mini_batch, 0, "mini batch");
DEFINE_double(sample_rate, 1.0, "sample rate");
DEFINE_bool(use_bias,  true, "if use bias");
DEFINE_bool(slave_update,  false, "if use bias");
DEFINE_double(bias, 1.0, "bias value");
DEFINE_int32(line_step, 100000, "log line step");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

using namespace ML;

struct SlaveThreadData {
    int seri;
    AdPredictorSlave* slave;
    std::vector<std::string> train_files;
};

class SlaveThread : public Common::Thread {
public:
    SlaveThread() {}
    virtual ~SlaveThread() {}
    void set(SlaveThreadData& data) {_data=data;}
    void addTrainFiles(std::string file) {_data.train_files.push_back(file);}
    virtual void run() {
        if (!_data.slave)
        {
            LOG_ERROR("Slave %d is NULL!", _data.seri);
            return;
        }
        if (_data.train_files.size() == 0)
        {
            LOG_ERROR("Slave %d Has No Train data", _data.seri);
        }
        for (size_t i=0; i<_data.train_files.size(); ++i)
        {
            LOG_INFO("Slave %d train file %s", _data.seri, _data.train_files[i].c_str());
            _data.slave->train(_data.train_files[i]);
        }
    }
private:
    SlaveThreadData _data;
};

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    ML::AdPredictorMaster model;
    model.init(FLAGS_init_mean, FLAGS_init_variance, FLAGS_beta, FLAGS_eps, FLAGS_max_fea_num, FLAGS_use_bias);
    int mini_batch = std::max(FLAGS_mini_batch, 1);
    LOG_INFO("mini_batch %d", mini_batch);
    int slave_num = std::max(FLAGS_slave_num, 1);
    std::vector<AdPredictorSlave> slaves(slave_num);
    std::vector<SlaveThread> threads(slave_num);
    for (int i=0; i<slave_num; ++i)
    {
        LOG_INFO("Init Slave %d", i+1);
        slaves[i].init(FLAGS_init_mean, FLAGS_init_variance, FLAGS_beta, FLAGS_eps, mini_batch, FLAGS_max_fea_num, FLAGS_use_bias, FLAGS_sample_rate, FLAGS_slave_update);
        slaves[i].set_seri(i+1);
        slaves[i].set_master(&model);
        SlaveThreadData data;
        data.slave = &(slaves[i]);
        data.seri = i+1;
        threads[i].set(data);
    }
    std::vector<std::string> train_files;
    Common::splitString(FLAGS_train_files, train_files, ',');
    for (size_t i=0; i<train_files.size(); ++i)
    {
        int slave_idx = i%slave_num;
        threads[slave_idx].addTrainFiles(train_files[i]);
        LOG_INFO("Add Train file %s to Slave %d", train_files[i].c_str(), slave_idx+1);
    }
    for (int i=0; i<slave_num; ++i)
    {
        LOG_INFO("Start Slave %d", i+1);
        threads[i].start();
    }
    for (int i=0; i<slave_num; ++i)
    {
        threads[i].join();
        LOG_INFO("End Slave %d", i+1);
    }
    model.save_model(FLAGS_model_file);
    return 0;
}