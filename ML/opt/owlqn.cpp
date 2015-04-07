#include "owlqn.h"
#include "Common/log.h"
#include <math.h>

namespace ML
{

void OWLQN::set_dim(int dim)
{
    _N = dim;
    vector<double> tmp(_N, 0);
    _w = tmp;
    _next_w = tmp;
    _dir = tmp;
    _grad = tmp;
    _next_grad = tmp;
    _steepest_dir = tmp;
    for (int i=0; i<=_M; ++i)
    {
        vector<double> tmp2(_N, 0);
        _Y.push_back(tmp2);
        _S.push_back(tmp2);
    }
    _alpha.resize(_M+1);
    _sy.resize(_M+1);
    _start = 0;
    _end = 0;
    _cur_iter = 0;
}

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
    if (_cur_iter == 0)
    {
        _model->grad_and_loss(_w, *_data, _grad, _loss);
    }
    double value = dotProduct(_grad, _grad);
    LOG_INFO("grad : %lf, dst : %lf", value, _error);
    if (_cur_iter > 0 && value < _error)
    {
        return true;
    }
    return false;
}

void OWLQN::optimize()
{
    // compute dir
    // compute grad
    while (_cur_iter < _max_iter)
    {
        LOG_INFO("-------------Iter %d start!----------------", _cur_iter);
		if (checkEnd())
		{
			LOG_INFO("Train finished After Iter %d!", _cur_iter);
            break;
		}
        LOG_INFO("Loss : %lf", _loss);
        updateDir();
        linearSearch();
        shiftState();
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
    LOG_DEBUG("start : %d, end : %d, M : %d", _start, _end, _M);
    int i = _end;
    while(i != _start)
    {
        i = (i-1+_M)%_M;
        LOG_DEBUG("i : %d, sy : %lf", i, _sy[i]);
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

double OWLQN::l1Loss()
{
	double loss = 0.0;
	_model->loss(_next_w, *_data, loss);
	if (_l1 > MinDoubleValue)
	{
		for (size_t i=0; i<_N; i++) {
			loss += fabs(_next_w[i])*_l1;
		}
	}
	return loss;
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
    double value = 0.0;

	while (true) {
        LOG_INFO("Linear search step : %lf", alpha);
		getNextPoint(alpha);
		value = l1Loss();
        LOG_DEBUG("Linear search value : %lf, old_value : %lf", value, oldValue);
		if (value <= oldValue + p * descDir * alpha) break;

		alpha *= backoff;
	}
}

void OWLQN::shiftState()
{
    _model->grad_and_loss(_next_w, *_data, _next_grad, _loss);
    for (size_t k=0; k<_w.size(); ++k)
    {
        _S[_end][k] = _next_w[k] - _w[k];
        _Y[_end][k] = _next_grad[k] - _grad[k];
        _w[k] = _next_w[k];
        _grad[k] = _next_grad[k];
    }
    _sy[_end] = dotProduct(_S[_end], _Y[_end]);
    _end = (_end+1) % _M;
    if (_cur_iter >= _M)
    {
        _start = (_start+1) % _M;
    }
}

}
