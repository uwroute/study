/**
 * @author: treesky - treesky.cn@gmail.com
 * @last modified 2014-04-09 17:31
 * @file fm_rec_algorithm.cpp
 * @description 
 */

#include "inc_fm.h"
#include <cmath>
#include "flag.h"

namespace ML
{

#define CHECK_MAP(map, key) \
    if (map.end() == map.find(key)) \
    {                               \
        map[key] = 0.0;             \
    }

IncFM::IncFM()
{
    // model param
    w0 = 0.0;
    // model train param
    _reg0 = 0.0;
    _regv = 0.0;
    _regw = 0.0;
    _L1 = 0.0;
    _num_factor = 0;
    _init_mean = 0;
    _init_stdev = 0.1;
    _train_method = SGD;
    // SGD
    _decay_rate = 1.0;
    // FTRL
    _alpha = 1.0;
    _beta = 0.05;
}

int IncFM::train(DoubleHashMap& instance, double label)
{
    switch (_train_method)
    {
        case SGD:
            sgd_train(instance, label);
            break;
        case FTRL:
            ftrl_train(instance, label);
            break
        default:
            return -1;
    }
    return 0;
}

int IncFM::sgd_train(DoubleHashMap& instance, double label)
{
    double p = predict(instance, true);
    double grad = -label * (1.0 - 1.0 / (1.0 + exp( -label * p)));

    w0 = w0 - cur_learn_rate(step) * (grad + reg0 * w0);
    step += 1;

    for (DoubleHashMap::const_iterator iter=instance.begin(); iter!=instance.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        double old_w = _w[fea_idx];
        _w[fea_idx] = old_w - cur_learn_rate(_step[fea_idx]) * (grad * iter->second + regw * old_w);
        for (uint32_t factor_idx = 0; factor_idx < _num_factor; ++factor_idx )
        {
            double v_grad = grad * iter->second * (_sum[factor_idx] - _v[fea_idx][factor_idx] * iter->second);
            double old_v = _v[fea_idx][factor_idx];
            v[fea_idx][factor_idx] = old_v - cur_learn_rate(_step[fea_idx]) * (v_grad + regv * old_v);
        }
        _step[fea_idx] += 1;
    }
    return 0;
}

int IncFM::ftrl_train(DoubleHashMap& instance, double label)
{
    /*double p = predict(instance, true);
    double grad = -label * (1.0 - 1.0 / (1.0 + exp( -label * p)));

    w0 = w0 - cur_learn_rate(step) * (grad + reg0 * w0);
    step += 1;

    for (DoubleHashMap::const_iterator iter=instance.begin(); iter!=instance.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        double old_w = _w[fea_idx];
        _w[fea_idx] = old_w - cur_learn_rate(_step[fea_idx]) * (grad * iter->second + regw * old_w);
        for (uint32_t factor_idx = 0; factor_idx < _num_factor; ++factor_idx )
        {
            double v_grad = grad * iter->second * (_sum[factor_idx] - _v[fea_idx][factor_idx] * iter->second);
            double old_v = _v[fea_idx][factor_idx];
            v[fea_idx][factor_idx] = old_v - cur_learn_rate(_step[fea_idx]) * (v_grad + regv * old_v);
        }
        _step[fea_idx] += 1;
    } // */
    return 0;
}

// y = w*x + sum(vi*vj*xi*xj)
double IncFM::predict(DoubleHashMap &ins, bool is_train)
{
    double result = 0;
    result += w0;
    for (DoubleHashMap::const_iterator iter=ins.begin(); iter!=ins.end(); ++iter)
    {
        uint64_t fea_idx = iter->first;
        if(_w.find(fea_idx) == _w.end())
        {
            if (!is_train)
            {
                continue;
            }
            else
            {
                _w[fea_idx] = 0.0;
            }
        }
        result += _w[fea_idx] * iter->second;
    }
    for (uint32_t factor_idx = 0; factor_idx < _num_factor; ++factor_idx)
    {
        _sum[factor_idx] = 0;
        _sum_square[factor_idx] = 0;
        for  (DoubleHashMap::const_iterator iter=ins.begin(); iter!=ins.end(); ++iter)
        {
            uint64_t fea_idx = iter->first;
            double fea_value = iter->second;
            if (_v.find() == _v.end())
            {
                if (!is_train)
                {
                    continue;
                }
                else
                {
                    std::vector<double> tmp(_num_factor);
                    gauss_vec(tmp);
                    _v[fea_idx] = tmp;
                }
            }
            double d = _v[fea_idx][factor_idx] * fea_value;
            _sum[factor_idx] += d;
            _sum_square[factor_idx] += d * d;
        }
        result += 0.5 * (_sum[factor_idx]*_sum[factor_idx] - _sum_square[factor_idx]);
    }// */
    return result;
}

int IncFM::save_model(std::string filename)
{
    std::ofstream model_out(filename.c_str());
    if (!model_out.is_open())
    {
        std::cout << "open model file to write failed!" << std::endl;
        return -1;
    }
    model_out << _reg0 << " " << _regv <<  " " << _regw << std::endl;
    model_out << w0 << " " << step << std::endl;
    model_out << _num_factor << std::endl;

    for (DoubleHashMap::const_ierator iter = _w.begin(); iter != _w.end(); iter++)
    {
        uint64_t fea_idx = iter->first;
        model_out << fea_idx << " " << _w[fea_idx] << " ";
        std::vector<double> fea_v = _v[fea_idx];
        for (size_t i=0; i<fea_v.size()-1; ++i)
        {
            std::cout << fea_v[i] << ",";
        }
        std::cout << fea_v[fea_v.size()-1];
        std::cout << " " << _step[fea_idx];
        std::cout << endl;
    }

    model_out.close();
    return 0;
}

int IncFM::cov_string_to_v(std::string& str, std::vector<double>& v)
{
    v.clear();
    size_t index = 0;
    size_t pos = str.find(",", index);
    while (pos != str.npos)
    {
        v.pushback(atof(str.substr(index, pos-index)));
        index = pos + 1;
        pos = str.find(",", index);
    }
    v.pushback(atof(str.substr(index, str.size()-index)));
    return 0;
}

int IncFM::load_model(std::string filename)
{
    std::ifstream model_in(filename.c_str());
    if (!model_in.is_open())
    {
        std::cout << "model file open to read failed!" << std::endl;
        return -1;
    }

    std::string line;
    getline(model_in, line);
    sscanf(line.c_str(), "%lf %lf %lf", &_reg0, &_regv, &_regw);
    getline(model_in, line);
    sscanf(line.c_str(), "%lf %u", &_w0, &step);
    getline(model_in, line);
    sscanf(line.c_str(), "%u", &_num_factor);

    char v_str[1024];
    uint64_t fea_idx = 0;
    step = 0;
    double w = 0.0;
    std::vector<double> tmp;
    getline(model_in, line);
    while (!model_in.eof())
    {
        sscanf(line.c_str(), "%lu %lf %s %u", &fea_idx, &w, v_str, &step);
        _w[fea_idx] = w;
        cov_string_to_v(v_str, tmp);
        _v[fea_idx] = tmp;
        _step[fea_idx] = step;
        getline(model_in, line);
    }

    model_in.close();
    return 0;
}

}
