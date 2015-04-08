#include "FTRL.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>

void FTRL::init(double a, double b, double l1, double l2, size_t max_fea_num)
{
    //_w.reserve(max_fea_num);
    //_z.reserve(max_fea_num);
    //_n.reserve(max_fea_num);
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

void FTRL::train(const DoubleHashMap& sample, double y)
{
    double h = predict(sample);
    double err = h-y;
    for (DoubleHashMap::const_iterator iter = sample.begin(); iter != sample.end(); ++iter)
    {
        CHECK_MAP(_n, iter->first);
        CHECK_MAP(_z, iter->first);
        CHECK_MAP(_w, iter->first);

		// grad|t
		// n|t = sum(grad*grad)|1:t
		// sigma = sqrt(n|t) - sqrt(n|t-1)/_alpha
		// z|t = z|t-1 + grad|t - sigma*w|t
        double grad = err*iter->second;
        double next_n = _n[iter->first] + grad*grad;
        double sigma = (sqrt(next_n) - sqrt(_n[iter->first]))/_alpha;
        _z[iter->first] += grad - sigma * _w[iter->first];
        _n[iter->first] = next_n;

		// |z|t|<_l1 : w|t = 0
		// other : w|t = -1/(1/lr + l2) * (z|t - sign(z|t)*lamda1)
		// z|t < -l1 : w|t > 0
		// z|t > l1 : w|t < 0
        if (fabs(_z[iter->first]) < _lamda1)
        {
            _w[iter->first] = 0.0;
        }
        else
        {
            double sign_z = _z[iter->first] > 0.0 ? 1.0 : -1.0;
            double lr = 1.0/((_beta + sqrt(_n[iter->first])/_alpha) + _lamda2);
            _w[iter->first] = -lr * (_z[iter->first] - sign_z * _lamda1);
        }
    }
}

double FTRL::predict(const DoubleHashMap& sample)
{
    double wx = 0.0;
    for (DoubleHashMap::const_iterator iter = sample.begin(); iter != sample.end(); ++iter)
    {
		if (_w.end() != _w.find(iter->first))
		{
			wx += _w[iter->first] * iter->second;
		}
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
