#ifndef __FTRL_H__
#define __FTRL_H__

#include <map>
#include <string>
#include <stdint.h>

class FTRL {
public:
    typedef std::map<uint64_t, double> DoubleHashMap;
    typedef std::map<uint64_t, uint32_t> Uint32HashMap;
    typedef std::map<uint64_t, uint64_t> Uint64HashMap;
public:
    FTRL() : _alpha(0.05), _beta(1.0), _lamda1(0.0), _lamda2(0.0) {}
    ~FTRL(){}
    void init(double a, double b, double l1, double l2, size_t max_fea_num = 10000*10000);
    void train(const DoubleHashMap& sample, double y);
    double predict(const DoubleHashMap& sample);
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

#endif
