#ifndef __FTRL_H__
#define __FTRL_H__

#include <map>
// #include <tr1/unordered_map>
#include <string>
#include <stdint.h>
#include "data/data.h"

namespace ML
{
// using std::tr1::unordered_map
class FTRL {
public:
    // typedef unordered_map<uint64_t, double> DoubleHashMap;
    typedef std::map<uint64_t, double> DoubleHashMap;
public:
    FTRL() : _alpha(0.05), _beta(1.0), _lamda1(0.0), _lamda2(0.0) {}
    ~FTRL(){}
    void init(double a, double b, double l1, double l2, size_t max_fea_num = 1000*10000);
    void train(const Feature* sample, double label);
    double predict(const Feature* sample);
    void save_model(const std::string& file);
    void load_model(const std::string& file);
public:
    void set_alpha(double alpha) {_alpha=alpha;}
    void set_beta(double beta) {_beta=beta;}
    void set_lamda1(double l1) {_lamda1=l1;}
    void set_lamda2(double l2) {_lamda2=l2;}
private:
    DoubleHashMap _w;
    DoubleHashMap _z;
    DoubleHashMap _n;
    double _alpha;
    double _beta;
    double _lamda1;
    double _lamda2;
};

}
#endif
