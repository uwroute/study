/**
 * @author: treesky - treesky.cn@gmail.com
 * @last modified 2014-04-09 17:31
 * @file fm_rec_algorithm.cpp
 * @description 
 */

#include "sgd_fm.h"
#include <cmath>
#include <fstream>
#include "Common/log.h"

namespace ML
{

#define CHECK_MAP(map, key) \
    if (map.end() == map.find(key)) \
    {                               \
        map[key] = 0.0;             \
    }

SgdFM::SgdFM()
{
    // model param
    _w0 = 0.0;
    // model train param
    _reg0 = 0.0;
    _regv = 0.0;
    _regw = 0.0;
    _num_factor = 0;
    _init_mean = 0;
    _init_stdev = 0.1;
    // SGD
    _decay_rate = 1.0;
    _step0 = 0;
}

int SgdFM::train(const Feature* sample, double label)
{
    if (_sum.size() != _num_factor)
    {
        std::vector<double> tmp(_num_factor, 0.0);
        _sum = tmp;
    }
    if (_sum_square.size() != _num_factor)
    {
        std::vector<double> tmp(_num_factor, 0.0);
        _sum_square = tmp;
    }
    double p = predict(sample, true);
    LOG_DEBUG("pre %lf", p);
    double grad = -label * (1.0 - 1.0 / (1.0 + exp( -label * p)));
    LOG_DEBUG("grad %lf", grad);

    _w0 = _w0 - cur_learn_rate(_step0) * (grad + _reg0 * _w0);
    _step0 += 1;

    LOG_DEBUG("%s", "train a sample");
    while (sample->index != -1)
    {
        uint64_t fea_idx = sample->index;
        double old_w = _w[fea_idx];
        _w[fea_idx] = old_w - cur_learn_rate(_step[fea_idx]) * (grad * sample->value + _regw * old_w);
        for (uint32_t factor_idx = 0; factor_idx < _num_factor; ++factor_idx )
        {
            double v_grad = grad * sample->value * (_sum[factor_idx] - _v[fea_idx][factor_idx] * sample->value);
            double old_v = _v[fea_idx][factor_idx];
            _v[fea_idx][factor_idx] = old_v - cur_learn_rate(_step[fea_idx]) * (v_grad + _regv * old_v);
            LOG_DEBUG("old %lf, new %lf, lr %lf", old_v,  _v[fea_idx][factor_idx], cur_learn_rate(_step[fea_idx]));
        }
        _step[fea_idx] += 1;
        sample++;
    }
    return 0;
}

// y = w*x + sum(vi*vj*xi*xj)
double SgdFM::predict(const Feature* ins, bool is_train)
{
    double result = 0;
    result += _w0;
    const Feature* sample = ins;
    while (sample->index != -1)
    {
        uint64_t fea_idx = sample->index;
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
        result += _w[fea_idx] * sample->value;
        sample++;
    }
    for (uint32_t factor_idx = 0; factor_idx < _num_factor; ++factor_idx)
    {
        _sum[factor_idx] = 0;
        _sum_square[factor_idx] = 0;
        sample = ins;
        while (sample->index != -1)
        {
            uint64_t fea_idx = sample->index;
            double fea_value = sample->value;
            if (_v.find(fea_idx) == _v.end())
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
            LOG_DEBUG("%lf, %lf", _sum[factor_idx], _sum_square[factor_idx]);
            double d = _v[fea_idx][factor_idx] * fea_value;
            _sum[factor_idx] += d;
            _sum_square[factor_idx] += d * d;
            sample++;
        }
        result += 0.5 * (_sum[factor_idx]*_sum[factor_idx] - _sum_square[factor_idx]);
    }// */
    return result;
}

int SgdFM::save_model(std::string filename)
{
    std::ofstream model_out(filename.c_str());
    if (!model_out.is_open())
    {
        std::cout << "open model file to write failed!" << std::endl;
        return -1;
    }
    model_out << _reg0 << " " << _regv <<  " " << _regw << std::endl;
    model_out << _w0 << " " << _step0 << std::endl;
    model_out << _num_factor << std::endl;

    for (DoubleHashMap::const_iterator iter = _w.begin(); iter != _w.end(); iter++)
    {
        uint64_t fea_idx = iter->first;
        model_out << fea_idx << " " << _w[fea_idx] << " ";
        model_out << _step[fea_idx];
        model_out << " ";
        std::vector<double> fea_v;
        if (_v.find(fea_idx) != _v.end())
        {
            fea_v = _v[fea_idx];
            for (size_t i=0; i<fea_v.size(); ++i)
            {
                model_out << fea_v[i] << ",";
            }
            if (fea_v.size() > 1)
            {
                model_out << fea_v[fea_v.size()-1];
            }
        }
        model_out << std::endl;
    }

    model_out.close();
    return 0;
}

int SgdFM::cov_string_to_v(std::string str, std::vector<double>& v)
{
    v.clear();
    size_t index = 0;
    size_t pos = str.find(",", index);
    while (pos != str.npos)
    {
        v.push_back(atof(str.substr(index, pos-index).c_str()));
        index = pos + 1;
        pos = str.find(",", index);
    }
    v.push_back(atof(str.substr(index, str.size()-index).c_str()));
    return 0;
}

int SgdFM::load_model(std::string filename)
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
    sscanf(line.c_str(), "%lf %u", &_w0, &_step0);
    getline(model_in, line);
    sscanf(line.c_str(), "%u", &_num_factor);
    _sum.resize(_num_factor);
    _sum_square.resize(_num_factor);

    char v_str[1024];
    uint64_t fea_idx = 0;
    uint32_t step = 0;
    double w = 0.0;
    std::vector<double> tmp;
    getline(model_in, line);
    while (!model_in.eof())
    {
        sscanf(line.c_str(), "%lu %lf %u %s", &fea_idx, &w, &step, v_str);
        _w[fea_idx] = w;
        if (_num_factor > 0)
        {
            cov_string_to_v(v_str, tmp);
            _v[fea_idx] = tmp;
        }
        _step[fea_idx] = step;
        getline(model_in, line);
    }

    model_in.close();
    return 0;
}

}
