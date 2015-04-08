#include "ftrl.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"

namespace ML {

void FTRL::init(double a, double b, double l1, double l2, size_t max_fea_num)
{
    // _w.reserve(max_fea_num);
    // _z.reserve(max_fea_num);
    // _n.reserve(max_fea_num);
    _alpha = a;
    _beta = b;
    _lamda1 = l1;
    _lamda2 = l2;
}

#define CHECK_MAP(map, key) \
    if (map.end() == map.find(key)) \
    {                               \
        map[key] = 0.0;             \
    }

void FTRL::train(const Feature* sample, double label)
{
    if (label < 0.5)
    {
        label = 0.0;
    }
    double h = predict(sample);
    double err = h-label;
    while (sample->index != -1)
    {
        CHECK_MAP(_n, sample->index);
        CHECK_MAP(_z, sample->index);
        CHECK_MAP(_w, sample->index);

        // grad|t
        // n|t = sum(grad*grad)|1:t
        // sigma = sqrt(n|t) - sqrt(n|t-1)/_alpha
        // z|t = z|t-1 + grad|t - sigma*w|t
        double grad = err * sample->value;
        double next_n = _n[sample->index] + grad*grad;
        double sigma = (sqrt(next_n) - sqrt(_n[sample->index]))/_alpha;
        _z[sample->index] += grad - sigma * _w[sample->index];
        _n[sample->index] = next_n;

        // |z|t|<_l1 : w|t = 0
        // other : w|t = -1/(1/lr + l2) * (z|t - sign(z|t)*lamda1)
        // z|t < -l1 : w|t > 0
        // z|t > l1 : w|t < 0
        if (fabs(_z[sample->index]) < _lamda1)
        {
            _w[sample->index] = 0.0;
        }
        else
        {
            double sign_z = _z[sample->index] > 0.0 ? 1.0 : -1.0;
            double lr = 1.0/((_beta + sqrt(_n[sample->index])/_alpha) + _lamda2);
            _w[sample->index] = -lr * (_z[sample->index] - sign_z * _lamda1);
        }
        sample++;
    }
}

double FTRL::predict(const Feature* sample)
{
    double wx = 0.0;
    while (sample->index != -1)
    {
        if (_w.end() != _w.find(sample->index))
        {
            wx += _w[sample->index] * sample->value;
        }
        sample++;
    }
    if (wx > 30.0)
    {
        return 1.0;
    }
    else if (wx < -30.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0/(1.0 + exp(-wx));
    }
}

void FTRL::save_model(const std::string& file)
{
    std::ofstream out_file(file.c_str());
    out_file << _alpha << std::endl;
    out_file << _beta << std::endl;
    out_file << _lamda1 << std::endl;
    out_file << _lamda2 << std::endl;
    out_file << _w.size() << std::endl;
    for (DoubleHashMap::const_iterator iter = _w.begin(); iter != _w.end(); ++iter)
    {
        out_file << iter->first << "\t" 
            << _w[iter->first] << "\t"
            << _z[iter->first] << "\t"
            << _n[iter->first] << std::endl;
    }
}

void FTRL::load_model(const std::string& file)
{
    std::ifstream infile(file.c_str());
    std::string line;
    // param
    getline(infile, line);
    _alpha = atof(line.c_str());
    getline(infile, line);
    _beta = atof(line.c_str());
    getline(infile, line);
    _lamda1 = atof(line.c_str());
    getline(infile, line);
    _lamda2 = atof(line.c_str());
    getline(infile, line);
    // int size = atoi(line.c_str());
    // _w.reserve(size);
    // _z.reserve(size);
    // _n.reserve(size);
    // w,z,n
    getline(infile, line);
    while (!infile.eof())
    {
        uint64_t fea = 0;
        double w=0.0, z=0.0, n=0.0;
        sscanf(line.c_str(), "%lu\t%lf\t%lf\t%lf", &fea, &w, &z, &n);
        _w[fea] = w;
        _z[fea] = z;
        _n[fea] = n;
        getline(infile, line);
    }
}

}
