#include "SparseBatchLR.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "Common/string_util.h"
#include "Common/log.h"

const double SparseBatchLR::MIN_DOUBLE = 1.0e-10;
// public interface
void SparseBatchLR::load_data_file(const std::string& file)
{
    std::ifstream infile(file.c_str());
    std::string line;
    // samples
    getline(infile, line);
    DoubleSparseVec vec;
    double label = 0.0;
    while (!infile.eof())
    {
        vec.clear();
        size_t max_fea_num = Common::toSample(line, vec, label);
        if (max_fea_num > 0)
        {
            label = std::max(0.0, label);
            _fea_num = std::max(_fea_num, max_fea_num);
            _data.push_back(vec);
            _label.push_back(label);
            _sample_num ++;
        }
        getline(infile, line);
    }
    LOG_INFO("FEA_NUM = %lu, SAMPLE_NUM = %lu\n", _fea_num, _sample_num);
    _loss = 0.0;
    _next_loss = 0.0;
    uint32_t real_fea_num = has_bias() ? _fea_num+1 : _fea_num;
    _grad = DoubleDenseVec(real_fea_num, 0.0);
    _w = DoubleDenseVec(real_fea_num, 0.0);
    _next_grad = DoubleDenseVec(real_fea_num, 0.0);
    _next_w = DoubleDenseVec(real_fea_num, 0.0);
    _d = DoubleDenseVec(real_fea_num, 0.0);
    switch (_train_method)
    {
        case TRAIN_METHOD_GD:
            break;
        case TRAIN_METHOD_BFGS:
            init_square_matrix(_H, real_fea_num);
            init_matrix(_next_H, real_fea_num, real_fea_num);
            break;
        case TRAIN_METHOD_OWLQN:
           init_matrix(_Y, _M+1, real_fea_num);
           init_matrix(_S, _M+1, real_fea_num);
           _q = DoubleDenseVec(real_fea_num, 0.0);
           _p = DoubleDenseVec(real_fea_num, 0.0);
           _a = DoubleDenseVec(_M + 1, 0.0);
           _b = DoubleDenseVec(_M + 1, 0.0);
           _rho = DoubleDenseVec(_M + 1, 0.0);
           break;
        default:
            break;
    }
}
void SparseBatchLR::load_model(const std::string& file)
{
    std::ifstream infile(file.c_str());
    // head
    std::string line;
    getline(infile, line);
    _fea_num = atoi(line.c_str());
    
    double w0 = 0.0;
    getline(infile, line);
    _bias = atof(line.c_str());
    sscanf(line.c_str(), "%lf %lf", &_bias, &w0);

    has_bias() ? _w.resize(_fea_num+1) : _w.resize(_fea_num);
    if (has_bias())
    {
        _w[_fea_num] = w0;
    }

    getline(infile, line);
    while (!infile.eof())
    {
        size_t idx = 0;
        double weight = 0.0;
        sscanf(line.c_str(), "%lu\t%lf", &idx, &weight);
        _w[idx] = weight;
        getline(infile, line);
    }
}
void SparseBatchLR::save_model(const std::string& file)
{
    std::ofstream outfile(file.c_str());
    outfile << _fea_num << std::endl;
    if (has_bias())
    {
        outfile << _bias << " " << _w[_fea_num] << std::endl;
    }
    else
    {
        outfile << _bias << " " << 0.0 << std::endl;
    }
    for (size_t i=0; i<_fea_num; ++i)
    {
        outfile << i << "\t" << _w[i] << std::endl;
    }
}

double SparseBatchLR::predict(const DoubleSparseVec& sample)
{
    return logistic(predict_wx(sample, _w));
}

void SparseBatchLR::train()
{
    _train_finish = false;
    for (uint32_t iter=0; iter < _max_iter; ++iter)
    {
        if (_train_finish)
        {
            LOG_INFO("%s", "-------Train Finish!--------");
            break;
        }
        LOG_INFO("--------Iter %u Start!--------", iter);
        if (train_once())
        {
            LOG_ERROR("--------Iter %u failed!--------", iter);
        }
        LOG_INFO("--------Iter %u End!--------", iter);
    }
}

// train process
int SparseBatchLR::train_once()
{
    switch (_train_method)
    {
    case TRAIN_METHOD_GD:
        GD();
        break;
    case TRAIN_METHOD_BFGS:
        BFGS();
        break;
    case TRAIN_METHOD_OWLQN:
        OWLQN();
        break;
    default:
        LOG_ERROR("Train Method if invalid : %d!", _train_method);
        return -1;
    }
    return 0;
}

// GD
void SparseBatchLR::GD()
{
    // compute now gradient
    if (!_use_pre)
    {
        batch_grad_and_loss(_w, _grad, _loss);
    }
    else if (_linear_search_method < Wolf)
    {
        batch_grad(_w, _grad);
        _loss = _next_loss;
    }
    else
    {   
        _grad = _next_grad;
        _loss = _next_loss;
    }
    _use_pre = false;
    double square_grad = sqrt(dot(_grad, _grad));
    LOG_INFO("now grad : %lf" , square_grad);
    LOG_INFO("now loss : %lf" , _loss);
    if (square_grad < _error)
    {
        LOG_INFO("now grad : %lf less than %lf" , square_grad, _error);
        _train_finish = true;
        return;
    }
    for (size_t i = 0; i < _grad.size(); ++i)
    {
        _d[i] = -1.0*_grad[i];
    }
    _cur_learn_rate = linear_search();
    _w = _next_w;
    LOG_INFO("linear search result : %lf", _cur_learn_rate);
}

// BFGS
void SparseBatchLR::BFGS()
{
    // compute now gradient
    if (!_use_pre)
    {
        batch_grad_and_loss(_w, _grad, _loss);
    }
    else
    {   
        _grad = _next_grad;
        _loss = _next_loss;
    }
    _use_pre = false;
    double square_grad = sqrt(dot(_grad, _grad));
    LOG_INFO("now grad : %lf" , square_grad);
    LOG_INFO("now loss : %lf" , _loss);
    if (square_grad < _error)
    {
        LOG_INFO("now grad : %lf less than %lf" , square_grad, _error);
        _train_finish = true;
        return;
    }
    for (size_t i = 0; i < _d.size(); ++i)
    {
        _d[i] = 0.0;
        for (size_t j=0; j < _d.size(); ++j)
        {
            _d[i] -= _H[i][j]*_grad[j];
        }
    }
    // linear_search
    _cur_learn_rate = linear_search();
    LOG_INFO("linear search result : %lf", _cur_learn_rate);
    // update _H
    if (_linear_search_method < Wolf)
    {
        batch_grad(_next_w, _next_grad);
    }
    DoubleDenseVec s(_d.size());
    DoubleDenseVec y(_d.size());
    for (size_t i=0; i<_d.size(); ++i)
    {
        s[i] = _next_w[i] - _w[i];
        y[i] = _next_grad[i] - _grad[i];
    }
    double sy = dot(s, y);
    for (size_t i=0; i<_d.size(); ++i)
    {
        for (size_t j=0; j<_d.size(); ++j)
        {
            _next_H[i][j] = 0.0;
            for (size_t k=0; k<_d.size(); ++k)
            {
                double unit = ((i==k) ? 1.0 : 0.0);
                _next_H[i][j] += (unit - s[i]*y[k]/sy)*_H[k][j];
            }
        }
    }
    for (size_t i=0; i<_d.size(); ++i)
    {
        for (size_t j=0; j<_d.size(); ++j)
        {
            _H[i][j] = 0.0;
            for (size_t k=0; k<_d.size(); ++k)
            {
                double unit = ((j==k) ? 1.0 : 0.0);
                _H[i][j] += _next_H[i][k]*(unit - y[k]*s[j]/sy);
            }
        }
    }
    for (size_t i=0; i<_d.size(); ++i)
    {
        for (size_t j=0; j<_d.size(); ++j)
        {
            _H[i][j] += s[i]*s[j]/sy;
        }
    }
    _w = _next_w;
}

// LBFGS
void SparseBatchLR::OWLQN()
{
    // compute now gradient
    if (!_use_pre)
    {
        batch_grad_and_loss(_w, _grad, _loss);
    }
    else
    {
        _grad = _next_grad;
        _loss = _next_loss;
    }
    _use_pre = false;
    double square_grad = sqrt(dot(_grad, _grad));
    LOG_INFO("now grad : %lf" , square_grad);
    LOG_INFO("now loss : %lf" , _loss);
    if (square_grad < _error)
    {
        LOG_INFO("now grad : %lf less than %lf" , square_grad, _error);
        _train_finish = true;
        return;
    }
    // two loop compute _d
    // a[i] = _S[i]*(I-Y[i+1] X S[i+1])...(I-Y[M] X S[M])*g
    // q = (I-Y[1] X S[1])...(I-Y[M] X S[M])*g
    // rho = S[i]*Y[i] i~[1, M]
    _q = DoubleDenseVec(_grad.size(), 0.0);
    _p = DoubleDenseVec(_grad.size(), 0.0);
    _a = DoubleDenseVec(_M + 1, 0.0);
    _b = DoubleDenseVec(_M + 1, 0.0);
    // _rho = DoubleDenseVec(_M + 1, 0.0);
    size_t index = _end;
    for (size_t i=0; i<_grad.size(); ++i)
    {
        if (is_positive(_reg1))
        {
            _next_grad[i] = virtual_grad(_w[i], _grad[i], _reg1);
            _q[i] = _next_grad[i];
        }
        else
        {
            _q[i] = _grad[i];
        }
    }
    while (index != _start)
    {
        if (index == 0)
        {
            index = _M;
        }
        else
        {
            index = (index - 1) % (_M + 1);
        }
        _a[index] = 0;
        for (size_t i=0; i<_grad.size(); ++i)
        {
            _a[index] += _S[index][i]*_q[i];
        }
        _a[index] /= _rho[index];
        for (size_t i=0; i<_grad.size(); ++i)
        {
            _q[i] -= _a[index]*_Y[index][i];
        }
    }
    // p = H*grad
    // b[i] = Y[i](I- S[i-1] X Y[i-1])...(I - S[1] X Y[1])*q
    for (size_t i=0; i<_grad.size(); ++i)
    {
        _p[i] = _q[i];
    }
    index = _start;
    while (index != _end)
    {
        _b[index] = 0;
        for (size_t i=0; i<_grad.size(); ++i)
        {
            _b[index] += _Y[index][i]*_p[i];
        }
        _b[index] /= _rho[index];
        for (size_t i=0; i<_grad.size(); ++i)
        {
            _p[i] += _S[index][i]*(_a[index] - _b[index]);
            // p[i] -= _S[index][i]*b[index];
        }
        index = (index + 1) % (_M + 1);
    }
    for (size_t i = 0; i < _grad.size(); ++i)
    {
        _d[i] = -1.0*_p[i];
        if (is_positive(_reg1))
        {
            if (_d[i]*_next_grad[i] >= 0.0)
            {
                _d[i] = 0.0;
            }
        }//*/
    }
    
    // linear_search
    time_t t1 = time(NULL);
    _cur_learn_rate = linear_search();
    time_t t2 = time(NULL);
    LOG_INFO("linear search result : %lf", _cur_learn_rate);
    LOG_DEBUG("linear search cost : %ld", t2-t1);
    // update _S _Y
    if (_linear_search_method < Wolf)
    {
        batch_grad(_next_w, _next_grad);
    }
    for (size_t i=0; i<_grad.size(); ++i)
    {
        _S[_end][i] = _next_w[i] - _w[i];
        _Y[_end][i] = _next_grad[i] - _grad[i];
    }
    _rho[_end] = dot(_S[_end], _Y[_end]);
    _end = (_end + 1) % (_M + 1);
    if (_end == _start)
    {
        _start = (_start + 1) % (_M + 1);
    }
    _w = _next_w;
}

// Linear search : A \ Wolf
double SparseBatchLR::linear_search()
{
    // init param
    double left = 0.0;
    double right = -1.0;
    const double alpha = 1.5; // increase rate
    double lamda = 1.0;       // try step
    const double p = 1.0e-4;     // p ~ (0, 0.5), condition
    const double sigma = 0.7; // sigma ~ (p, 1)
    // now grad and loss, grad*d (grad*d = -grad*H*grad, because H is posive matrix, so grad*d must be negative < 0)
    double gd = 0.0;
    if (_train_method == TRAIN_METHOD_OWLQN && _reg1 > 0.0)
    {
        gd = virtual_dot(_w, _d, _grad, _reg1);
        LOG_DEBUG("gd = %lf", gd);
    }
    else
    {
        gd = dot(_d, _grad);
    }

    // init next 
    bool is_finish = false;
    double next_gd = 0.0;
    // stop codition : satisfy A or Wolf or region is samll
    while (!is_finish && fabs(right - left) > 1e-5)
    {
        // compute next_w, next_grad, next_loss, next grad*d
        for (size_t i=0; i<_w.size(); ++i)
        {
            _next_w[i] = _w[i] + lamda*_d[i];
            if (_train_method == TRAIN_METHOD_OWLQN && _reg1 > 0.0)
            {
                if (_next_w[i] * _w[i] < 0.0)
                {
                    _next_w[i] = 0.0;
                }
            }
        }
        // only wolf need next_grad
        switch (_linear_search_method)
        {
        case Wolf:
        case StrongWolf:
            batch_grad_and_loss(_next_w, _next_grad, _next_loss);
            next_gd = dot(_d, _next_grad);
            break;
        case Simple:
        case Armijo:
        default:
            batch_loss(_next_w, _next_loss);
            break;
        }
        
        // first condition : loss is decay
        if (_next_loss <= (_loss + lamda*p*gd))
        {
            // second condition
            // Simple : none
            // Armijo : loss decay can't be too large
            // Wolf : next grad must be less than now grad in same side, but other side no constant
            // StrongWolf : next grad must be less than now grad for two side, 
            switch (_linear_search_method)
            {
            case Simple:
                is_finish = true;
                break;
            case Armijo:
                is_finish = (_next_loss >= (_loss+lamda*(1-p)*gd) ? true : false);
                break;
            case Wolf:
                is_finish = (next_gd >= sigma*gd ? true : false);
                break;
            case StrongWolf:
                is_finish = (fabs(next_gd) <= -sigma*gd ? true : false);
                break;
            default:
                is_finish = true;
                break;
            }
            // not satisfy second condition, step is too small
            if (!is_finish)
            {
                left = lamda;
                // first 
                if (right < 0)
                {
                    lamda *= alpha;
                }
                else
                {
                    // new region
                    lamda = (left + right)/2;
                }
            } // */
        }
        else
        {
            // not satisfy first condition, step is too large
            right = lamda;
            lamda = (left + right)/2;
        }
    }
    _use_pre = true;
    return lamda;
}

// private util function for train
void SparseBatchLR::batch_grad(const DoubleDenseVec& w, DoubleDenseVec& grad)
{
    time_t t1 = time(NULL);
    for (size_t j=0; j<grad.size(); ++j)
    {
        grad[j] = 0.0;
    }
    for (size_t i=0; i<_sample_num; ++i)
    {
        add_instance(w, _data[i], _label[i], grad);
    }
    if (_reg2 > 0.0)
    {
        for (size_t j=0; j<grad.size(); ++j)
        {
            grad[j] += _reg2*w[j];
        }
    }
    time_t t2 = time(NULL);
    LOG_DEBUG("batch grad time cost : %lu", t2-t1);
}

void SparseBatchLR::batch_loss(const DoubleDenseVec& w, double& loss)
{
    time_t t1 = time(NULL);
    loss = 0.0;
    for (size_t i=0; i<_sample_num; ++i)
    {
        add_instance(w, _data[i], _label[i], loss);
    }
    if (_reg2 > 0.0)
    {
        loss += 0.5*_reg2*dot(w, w);
    }
    time_t t2 = time(NULL);
    LOG_DEBUG("batch loss time cost : %lu", t2-t1);
}

void SparseBatchLR::batch_grad_and_loss(const DoubleDenseVec& w, DoubleDenseVec& grad, double& loss)
{
    time_t t1 = time(NULL);
    for (size_t j=0; j<grad.size(); ++j)
    {
        grad[j] = 0.0;
    }
    loss = 0.0;
    for (size_t i=0; i<_sample_num; ++i)
    {
        add_instance(w, _data[i], _label[i], grad, loss);
    }
    if (_reg2 > 0.0)
    {
        for (size_t j=0; j<grad.size(); ++j)
        {
            grad[j] += _reg2*w[j];
        }
        loss += 0.5*_reg2*dot(w, w);
    }
    time_t t2 = time(NULL);
    LOG_DEBUG("batch grad and loss time cost : %lu", t2-t1);
}

void SparseBatchLR::add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, double& loss)
{
    double wx = predict_wx(sample, w);
    loss += log_loss(wx, label);
}
void SparseBatchLR::add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, DoubleDenseVec& grad)
{
    double wx = predict_wx(sample, w);
    double h = logistic(wx);
    double y = label;
    double err = h - y;
    for (DoubleSparseVec::const_iterator iter = sample.begin(); iter != sample.end(); ++iter)
    {
        grad[iter->first] += err*iter->second;
    }
    if (has_bias())
    {
        grad[_fea_num] += err*_bias;
    }
}
void SparseBatchLR::add_instance(const DoubleDenseVec& w, const DoubleSparseVec& sample, const double& label, DoubleDenseVec& grad, double& loss)
{
    double wx = predict_wx(sample, w);
    double h = logistic(wx);
    double y = label;
    double err = h - y;
    for (DoubleSparseVec::const_iterator iter = sample.begin(); iter != sample.end(); ++iter)
    {
        grad[iter->first] += err*iter->second;
    }
    if (has_bias())
    {
        grad[_fea_num] += err*_bias;
    }
    loss += log_loss(wx, label);
    // LOG_TRACE("label = %lf, pre = %lf, loss = %lf\n", label, h, log_loss(wx, label));
}

void SparseBatchLR::init_matrix(std::vector<DoubleDenseVec>& matrix, size_t M, size_t N)
{
    matrix.resize(M);
    for (size_t i=0; i<M; ++i)
    {
        matrix[i] = DoubleDenseVec(N, 0.0);
    }
}

void SparseBatchLR::init_square_matrix(std::vector<DoubleDenseVec>& matrix, size_t N)
{
    matrix.resize(N);
    for (size_t i=0; i<N; ++i)
    {
        matrix[i] = DoubleDenseVec(N, 0.0);
        matrix[i][i] = 1.0;
    }
}

double SparseBatchLR::virtual_dot(const DoubleDenseVec& w, const DoubleDenseVec& dir, const DoubleDenseVec& grad, double reg1)
{
    double res = 0.0;
    for (DoubleDenseVec::const_iterator iw = w.begin(), ig = grad.begin(), id = dir.begin();
        iw != w.end() && ig != grad.end() && id != dir.end(); ++iw, ++ig, ++id)
    {
        if (is_negative(*id) || is_positive(*id))
        {
            res += virtual_gd(*iw, *ig, *id, reg1);
        }
    }
    return res;
}
