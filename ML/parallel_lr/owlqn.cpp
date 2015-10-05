/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-09-04 22:06
#
# Filename: owlqn.cpp
#
# Description: 
#
=============================================================================*/

#include "owlqn.h"
#include "Common/log.h"
#include <math.h>

namespace ML
{

void OWLQN::init()
{
    if (_N <= 0)
    {
        LOG_ERROR("Dim[%d] must be large than 0!", _N);
        return -1;
    }
    // model param
    vector<double> tmp(_N, 0);
    _w.swap(tmp);
    _next_w = _w;
    _dir = _w;
    _grad = _w;
    _next_grad = _w;
    // LBFGS param
    for (int i=0; i<=_M; ++i)
    {
        _Y.push_back(_w);
        _S.push_back(_w);
    }
    _alpha.resize(_M+1);
    _sy.resize(_M+1);
    _start = 0;
    _end = 0;
    // train control param
    _cur_iter = 0;
    return 0;
}
// util function
double OWLQN::dotProduct(vector<double>& x, vector<double>& y)
{
    double value = 0.0;
    for (size_t i=0; i<x.size(); ++i)
    {
        value += x[i]*y[i];
    }
    return value;
}
void OWLQN::add(vector<double>& out, const vector<double>& in)
{
    for (size_t i=0; i<out.size(); ++i)
    {
        out[i] += in[i];
    }
}

void OWLQN::addScale(vector<double>& out, const vector<double>& x, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] += scale*x[i];
    }
}

void OWLQN::addScaleInto(vector<double>& out, const vector<double>& x, const vector<double>& y, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] = x[i] + scale*y[i];
    }
}

void OWLQN::scale(vector<double>& out, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] *= scale;
    }
}
	
void OWLQN::scaleInto(vector<double>& out, const vector<double>& x, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] = x[i]*scale;
    }
}

bool OWLQN::checkEnd()
{
    double value = dotProduct(_grad, _grad);
    LOG_INFO("grad : %lf, dst : %lf", value, _error);
    if (_cur_iter > 0 && value < _error)
    {
        return true;
    }
    return false;
}

void OWLQN::calc_grad()
{
    grad_status.set_state(CALC_GRAD);
    grad_status.init_done_num();
    read_status.set_state(READ_START);
    while (grad_status.done_num() != GRAD_THREAD_NUM)
    {
        usleep(1);
    }
    read_status.set_state(READ_IDLE);
    grad_status.set_state(CALC_IDLE);
}

void OWLQN::calc_loss()
{
    grad_status.set_state(CALC_LOSS);
    grad_status.init_done_num();
    read_status.set_state(READ_START);
    while (grad_status.done_num() != GRAD_THREAD_NUM)
    {
        usleep(1);
    }
    read_status.set_state(READ_IDLE);
    grad_status.set_state(CALC_IDLE);
}

void OWLQN::calc_grad_and_loss()
{
    grad_status.set_state(CALC_GRAD_AND_LOSS);
    grad_status.init_done_num();
    read_status.set_state(READ_START);
    while (grad_status.done_num() != GRAD_THREAD_NUM)
    {
        usleep(1);
    }
    read_status.set_state(READ_IDLE);
    grad_status.set_state(CALC_IDLE);
}

void OWLQN::optimize()
{
    // compute dir
    // compute grad
    calc_grad_and_loss();
    while (_cur_iter < _max_iter)
    {
        if (checkEnd())
        {
            LOG_INFO("Train finished After Iter %d!", _cur_iter);
            break;
        }
        LOG_INFO("Loss : %lf", _loss);
        updateDir(); // parallel
        linearSearch(); // parallel
        shiftState(); // single
        LOG_INFO("-------------Iter %d end!------------------", _cur_iter);
        _cur_iter ++;
    }
    _model->set_param(_w);
}

void OWLQN::updateDir()
{
    LOG_DEBUG("%s", "Update Dir Start!");
    LOG_DEBUG("%s", "Compute steepest desc dir!");
    makeSteepestDescDir();
    LOG_DEBUG("%s", "Compute newton dir!");
    mapDirByInverseHessian();
    LOG_DEBUG("%s", "Fix dir sign!");
    fixDirSign();
    LOG_DEBUG("%s", "Update Dir End!");
}

// compute dir = -grad with l1
void OWLQN::makeSteepestDescDir()
{
	if (_l1 > MinDoubleValue)
    {
        for (size_t i=0; i<_N; ++i)
        {
            if (_w[i] > MinDoubleValue)
            {
                _dir[i] = -1.0*(_grad[i] + _l1);
            }
            else if (_w[i] < -1.0*MinDoubleValue)
            {
                _dir[i] = -1.0*(_grad[i] - _l1);
            }
            else
            {
                double l_grad = _grad[i]-_l1;
                double r_grad = _grad[i]+_l1;
                if (r_grad < -1.0*MinDoubleValue)
                {
                    _dir[i] = -1.0*r_grad;
                }
                else if (l_grad > MinDoubleValue)
                {
                    _dir[i] = -1.0*l_grad;
                }
                else
                {
                    _dir[i] = 0.0;
                }
            }
        }
    }
    else
    {
        for (size_t i=0; i<_N; ++i)
        {
            _dir[i] = -1.0*_grad[i];
        }
    }
    for (size_t i=0; i<_N; ++i)
    {
        _steepest_dir[i] = _dir[i];
    }
}

void OWLQN::mapDirByInverseHessian()
{
    // has no history info, dir = steepest_dir
    if (_cur_iter == 0)
    {
        return;
    }
    // two loop lbfgs
    // Hk+1 = G-1
    // Hk+1 * yk = sk
    // Hk+1 = (I-sy'/s'y)*Hk*(I-ys'/s'y) + ss'/s'y
    // dir = -Hk+1*grad
    // one loop, compute q[i] = 
    int i = _end;
    while(i != _start)
    {
        i = (i-1+_M)%_M;
        _alpha[i] = -1.0*dotProduct(_S[i], _dir) / _sy[i];
        addScale(_dir, _Y[i], _alpha[i]);
    }
    // H0 = I*(s'y/y'y)
    i = (_end-1+_M)%_M;
    double yy = _sy[i]/dotProduct(_Y[i], _Y[i]);
    LOG_DEBUG("init H = %lf * I", yy);
    scale(_dir, yy);
    // second loop
    i = _start;
    while (i != _end)
    {
        _beta = dotProduct(_Y[i], _dir) / _sy[i];
		addScale(_dir, _S[i], -_beta-_alpha[i]);
        i = (i+1) % _M;
    }
}

void OWLQN::fixDirSign()
{
	// fix dir sign
    // check dir direction, same direction with l1 grad
	if (_l1 > MinDoubleValue)
	{	
		for (size_t i=0; i<_N; ++i)
		{
			if (_dir[i]*_steepest_dir[i] <= 0)
			{
				_dir[i] = 0.0;
			}
		}
	}
}

double OWLQN::checkDir()
{
	double value = 0.0;
	if (_l1 < MinDoubleValue)
	{
		value = dotProduct(_dir, _grad);
	}
	else
	{	
		for (size_t i=0; i<_N; ++i)
		{
			value += -1.0*_dir[i]*_steepest_dir[i];
		}
	}
	return value;
}

void OWLQN::getNextPoint(double alpha) {
	addScaleInto(_next_w, _w, _dir, alpha);
	if (_l1 > MinDoubleValue) {
		for (size_t i=0; i<_N; i++) {
			if (_w[i] * _next_w[i] < -1.0*MinDoubleValue) {
				_next_w[i] = 0.0;
			}
		}
	}
}

double OWLQN::l1Loss(const vector<double>& w, const double loss)
{
	double l1Loss = loss;
	if (_l1 > MinDoubleValue)
	{
		for (size_t i=0; i<_N; i++) {
			l1Loss += fabs(w[i])*_l1;
		}
	}
	return l1Loss;
}

double OWLQN::l2Loss(const vector<double>& w, const double loss)
{
    double l2Loss = loss;
    if (_l2 > MinDoubleValue)
    {
        for (size_t i=0; i<_N; i++) {
            l2Loss += w[i]*w[i]*_l2*0.5;
        }
    }
    return l2Loss;
}

void OWLQN::linearSearch()
{
    // check is _dir down dir, so Hk+1 should be positive
    // l1grad * _dir should be less than 0
    double descDir = checkDir();
	if (descDir > 0.0)
	{
		LOG_ERROR("%s", "Dir is not desc direction !");
		exit(1);
	}
    LOG_DEBUG("descDir : %lf", descDir);
	
	double alpha = 1.0;
	double backoff = 0.5;
	if (_cur_iter == 0) {
		double normDir = sqrt(dotProduct(_dir, _dir));
		alpha = (1 / normDir);
		backoff = 0.1;
    }

	const double p = 1e-4;
	double oldValue = _loss;
    double& value = _loss;

	while (true) {
        LOG_INFO("Linear search step : %lf", alpha);
		getNextPoint(alpha);
        double sample_loss = 0.0;
        calc_loss();
		value = l1Loss();
        LOG_DEBUG("Linear search value : %lf, old_value : %lf", value, oldValue);
		if (value <= oldValue + p * descDir * alpha) break;

		alpha *= backoff;
	}
}

void OWLQN::shiftState()
{
    for (size_t k=0; k<_w.size(); ++k)
    {
        _S[_end][k] = _next_w[k] - _w[k];
        _Y[_end][k] = _next_grad[k] - _grad[k];
    }
    _sy[_end] = dotProduct(_S[_end], _Y[_end]);

    _w.swap(_next_w);
    _grad.swap(_next_grad);

    _end = (_end+1) % _M;
    if (_cur_iter >= _M)
    {
        _start = (_start+1) % _M;
    }
}

int OWLQN::caluc_space()
{
    int space = 0;
    space += sizeof(size_t); // _N
    space += _N*sizeof(double); // _w
    space += _N*sizeof(double); // _next_w
    space += _N*sizeof(double); // _dir
    space += _N*sizeof(double); // _grad
    space += _N*sizeof(double); // _next_grad
    space += sizeof(void*); // _steepest_dir
    space += (_M+1)*sizeof(double); // _alpha
    space += (_M+1)*sizeof(double); // _sy
    space += sizeof(double); // _beta
    space += (_M+1)*_N*sizeof(double); // _Y
    space += (_M+1)*_N*sizeof(double); // _S
    space += sizeof(int); // _start
    space += sizeof(int); // _end
    space += sizeof(void*); // _model
    space += sizeof(void*); // _data
    space += sizeof(double); // _l1
    space += sizeof(int); // _max_iter
    space += sizeof(int); // _cur_iter
    space += sizeof(double); // _error
    return space;
}

}