#include <owlqn.h>

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

void addScale(vector<double>& out, const vector<double>& x, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] += scale*x[i];
    }
}

void addScaleInto(vector<double>& out, const vector<double>& x, const vector<double>& y, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] = x + scale*y[i];
    }
}

void scale(vector<double>& out, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] *= scale;
    }
}
	
void scaleInto(vector<double>& out, const vector<double>& x, const double scale)
{
	for (size_t i=0; i<out.size(); ++i)
    {
        out[i] = x[i]*scale;
    }
}

void OWLQN::optimize()
{
    // compute dir
    // compute grad
    while (_cur_iter < _max_iter)
    {
		if (checkEnd())
		{
			LOG_INFO("Train finished After Iter %d!", _cur_iter);
		}
        _cur_iter ++;
        updateDir();
        linearSearch();
        shiftState();
    }
}

void OWLQN::updateDir()
{
	makeSteepestDescDir();
	mapDirByInverseHessian();
	fixDirSign();
}

void OWLQN::makeSteepestDescDir()
{
    model->grad_and_loss(_w, *_data, _grad, _loss);
	if (_l1>0)
    {
        for (int i=0; i<_N; ++i)
        {
            if (_w[i] > MinDoubleValue)
            {
                _dir[i] = -1.0*(_grad[i] + _l1)
            }
            else if (_w[i] < -1.0*MinDoubleValue)
            {
                _dir[i] = -1.0*(_grad[i] - _l1)
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
        for (int i=0; i<_N; ++i)
        {
            _dir[i] = -1.0*_grad[i];
        }
    }
}

void OWLQN::mapDirByInverseHessian()
{
    // two loop lbfgs
    // Hk+1 = G-1
    // Hk+1 * yk = sk
    // Hk+1 = (I-sy'/s'y)*Hk*(I-ys'/s'y) + ss'/s'y
    // dir = -Hk+1*grad
    // one loop, compute q[i] = 
    int i = _end;
    while(i != _start)
    {
        i = (i-1)%_M;
        _sy[i] = dotProduct(_S[i], _Y[i]);
        _alpha[i] = dotProduct(_S[i], _dir) / _sy[i];
		addScale(_dir, _Y[i], -alpha[i]);
    }
    // H0 = I*(s'y/y'y)
    i = (_end-1)%M;
    double yy = _sy[i]/dotProduct(_Y[i], _Y[i]);
	scale(_dir, yy);
    // second loop
    i = _start;
    while (i != _end)
    {
        _beta = dotProduct(_Y[i], _dir) / _sy[i];
		addScale(_dir, _S[i], -_beta-_alpha[i]);
        i = (i+1)%M;
    }
}

void fixDirSign()
{
	// fix dir sign
    // check dir direction, same direction with l1 grad
	if (_l1 > MinDoubleValue)
	{	
		for (int i=0; i<_N; ++i)
		{
			double steepest_dir = 0.0;
			if (_w[i] > MinDoubleValue)
            {
                steepest_dir = -1.0*(_grad[i] + _l1);
            }
            else if (_w[i] < -1.0*MinDoubleValue)
            {
                steepest_dir = -1.0*(_grad[i] - _l1);
            }
            else
            {
                double l_grad = _grad[i]-_l1;
                double r_grad = _grad[i]+_l1;
                if (r_grad < -1.0*MinDoubleValue)
                {
                    steepest_dir = -r_grad;
                }
                else if (l_grad > MinDoubleValue)
                {
                    steepest_dir = -l_grad;
                }
                else
                {
                    steepest_dir = 0.0;
                }
            }
			if (_dir[i]*steepest_dir <= 0)
			{
				_dir[i] = 0.0;
			}
		}
	}
}

double OWLQN::checkDir()
{
	double value = 0.0;
	if (_l1 < -1.0*MinDoubleValue)
	{
		value = dotProduct(_dir, _grad);
	}
	else
	{	
		for (int i=0; i<_N; ++i)
		{
			double l1_grad = 0.0;
			if (_w[i] > MinDoubleValue)
            {
                l1_grad = _grad[i] + _l1;
            }
            else if (_w[i] < -1.0*MinDoubleValue)
            {
                l1_grad = _grad[i] - _l1;
            }
            else
            {
                double l_grad = _grad[i]-_l1;
                double r_grad = _grad[i]+_l1;
                if (r_grad < -1.0*MinDoubleValue)
                {
                    l1_grad = r_grad;
                }
                else if (l_grad > MinDoubleValue)
                {
                    l1_grad = l_grad;
                }
                else
                {
                    l1_grad = 0.0;
                }
            }
			value += _dir[i]*l1_grad;
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
	model->loss(_next_w, *_data, loss);
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
	
	double alpha = 1.0;
	double backoff = 0.5;
	if (iter == 1) {
		//alpha = 0.1;
		//backoff = 0.5;
		double normDir = sqrt(dotProduct(_dir, _dir));
		alpha = (1 / normDir);
		backoff = 0.1;
	}

	const double p = 1e-4;
	double oldValue = _loss;

	while (true) {
		getNextPoint(alpha);
		value = l1Loss();

		if (value <= oldValue + p * descDir * alpha) break;

		alpha *= backoff;
	}
}

void OWLQN::shiftState()
{
    for (size_t k=0; k<_w.size(); ++i)
    {
        _S[_end][k] = _next_w[k] - _w[k];
        _Y[_end][k] = _next_grad[k] - _grad[k];
    }
    _end = (_end+1) % M;
    if (_cur_iter > M)
    {
        _start = (_start+1) % M;
    }
}
