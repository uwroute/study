
#ifndef _SPARSE_LR_
#define _SPARSE_LR_

#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <stdint.h>
#include "ML/Common/log.h"

typedef std::vector<double> DoubleDenseVec;
typedef std::map<uint64_t, double> DoubleSparseVec;

class SparseBatchLR {
    enum TRAIN_METHOD
    {
        TRAIN_METHOD_GD = 0,
        TRAIN_METHOD_BFGS = 1,
        TRAIN_METHOD_OWLQN = 2
    };
    enum LINEAR_SEARCH_METHOD
    {
        Simple = 0,
        Armijo = 1,
        Wolf = 2,
        StrongWolf = 3, 
    };
private:
    // model param
    DoubleDenseVec _w;
    // train param
    double _max_iter;
    double _error;
    double _reg1;
    double _reg2;
    size_t _M;
    double _bias;
    double _learn_rate;
    // train process data
    double _loss;
    DoubleDenseVec _grad;
    DoubleDenseVec _next_w;
    DoubleDenseVec _next_grad;
    double _next_loss;
    // GD
    DoubleDenseVec _d;
    // BFGS
    std::vector<DoubleDenseVec> _H;
    std::vector<DoubleDenseVec> _next_H;
    // LBFGS
    std::vector<DoubleDenseVec> _S;
    std::vector<DoubleDenseVec> _Y;
    DoubleDenseVec _q;
    DoubleDenseVec _p;
    DoubleDenseVec _a;
    DoubleDenseVec _b;
    DoubleDenseVec _rho;
    size_t _start;
    size_t _end;
    int _train_method;
    int _linear_search_method;
    double _cur_learn_rate;
    bool _use_pre;
    bool _train_finish;
    // data
    std::vector<DoubleSparseVec> _data;
    DoubleDenseVec _label;
    size_t _fea_num;
    size_t _sample_num;
public:
    SparseBatchLR() : _max_iter(0), _error(1.0e-5), _reg1(0.0), _reg2(0.0), _M(0), _bias(0), _learn_rate(1.0),
    _start(0), _end(0), _train_method(0), _linear_search_method(0), _cur_learn_rate(0.0), _use_pre(false),
    _train_finish(false), _fea_num(0), _sample_num(0) {}
    ~SparseBatchLR(){}
    void train();
    double predict(const DoubleSparseVec& sample);
    void load_data_file(const std::string& file);
    void load_model(const std::string& file);
    void save_model(const std::string& file);
public:
    // set train param
    void set_max_iter(uint32_t max_iter) { _max_iter = max_iter; }
    void set_error(double error) {_error = error;}
    void set_reg1(double l1) {_reg1 = l1;}
    void set_reg2(double l2) {_reg2 = l2;}
    void set_M(size_t M) {_M=M;}
    void set_bias(double bias) {_bias=bias;}
    void set_train_method(int method) {_train_method=method;}
    void set_linear_search_method(int method) {_linear_search_method = method;}
    void set_learn_rate(double lr) {_learn_rate = lr;}
private:
    bool has_bias() const {return _bias>1.0e-5;}
    void get_cur_learnrate();
    void batch_grad(const DoubleDenseVec& w, DoubleDenseVec& grad);
    void batch_loss(const DoubleDenseVec& w, double& loss);
    void batch_grad_and_loss(const DoubleDenseVec& w, DoubleDenseVec& grad, double& loss);
private:
    int train_once();
private:
    // opt method
    void GD();
    void BFGS();
    void OWLQN();
    double linear_search();
private:
    // util func
    const static double MIN_DOUBLE = 1.0e-10;
    bool is_negative(double x) {return x < -MIN_DOUBLE;}
    bool is_positive(double x) {return x > MIN_DOUBLE;}
    void init_matrix(std::vector<DoubleDenseVec>& matrix, size_t M, size_t N);
    void init_square_matrix(std::vector<DoubleDenseVec>& matrix, size_t N);
    double dot(const DoubleDenseVec& v1, const DoubleDenseVec& v2);
    double predict_wx(const DoubleSparseVec& sample, const DoubleDenseVec& w);
    double logistic(double wx);
    double log_loss(double wx, double label);
    double virtual_grad(double x, double grad, double reg1);
    double virtual_gd(double x, double grad, double dir, double reg1);
    double virtual_dot(const DoubleDenseVec& w, const DoubleDenseVec& grad, const DoubleDenseVec& dir, double reg1);
    void add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, double& loss);
    void add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, DoubleDenseVec& grad);
    void add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, DoubleDenseVec& grad, double& loss);
};

inline double SparseBatchLR::log_loss(double wx, double label)
{
    double loss = 0.0;
    if (label > 0.5)
    {
        wx *= -1.0;
    }
    if (wx > 30)
    {
        loss += wx;
    }
    else if (wx > -30)
    {
        loss += log(1+exp(wx));
    }
    return loss;
}
inline double SparseBatchLR::logistic(double wx)
{
    if (wx > 30)
    {
        return 1.0;
    }
    if (wx < -30)
    {
        return 0.0;
    }
    return 1.0/(1.0 + exp(-wx));
}
inline double SparseBatchLR::dot(const DoubleDenseVec& v1, const DoubleDenseVec& v2)
{
    double res = 0.0;
    for (DoubleDenseVec::const_iterator iter1 = v1.begin(), iter2 = v2.begin(); 
        iter1 != v1.end() && iter2 != v2.end(); ++iter1, ++iter2)
    {
        res += (*iter1)*(*iter2);
    }
    return res;
}
inline double SparseBatchLR::predict_wx(const DoubleSparseVec& sample, const DoubleDenseVec& w)
{
    double wx = 0.0;
    for (DoubleSparseVec::const_iterator iter = sample.begin(); iter != sample.end(); ++iter)
    {
        wx += iter->second * w[iter->first];
    }
    if (has_bias())
    {
        wx += _bias*w[_fea_num];
    }
    return wx;
}

inline double SparseBatchLR::virtual_grad(double x, double grad, double reg1)
{
    if (is_positive(x))
    {
        return grad + reg1;
    }
    else if (is_negative(x))
    {
        return grad - reg1;
    }
    else
    {
        if (fabs(grad) <= reg1)
        {
            return 0;
        }
        else if (is_negative(grad))
        {
            return grad + reg1;
        }
        else
        {
            return grad - reg1;
        }
    }
}

inline double SparseBatchLR::virtual_gd(double x, double grad, double dir, double reg1)
{
    if (is_positive(x))
    {
        return dir*(grad + reg1);
    }
    else if (is_negative(grad))
    {
        return dir*(grad - reg1);
    }
    else if (is_negative(dir))
    {
        return dir*(grad - reg1);
    }
    else if (is_positive(dir))
    {
        return dir*(grad + reg1);
    }
    else
    {
        return 0.0;
    }
}

#endif
