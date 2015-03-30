#include <owlqn.h>

void OWLQN::optimize()
{
    // compute dir
    // compute grad
    while (_cur_iter < _max_iter)
    {
        _cur_iter ++;
        grad();
        dir();
        linearSearch();
        shift();
    }
}

void OWLQN::grad()
{
    model->grad(_w, *_data, _grad);
}

void OWLQN::dir()
{
    for (int i=0; i<_N; ++i)
    {
        _dir[i] = _grad[i];
    }
    // l1 so virtual grad
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
                if (r_grad < 0.0)
                {
                    _dir[i] = -1.0*r_grad;
                }
                else if (l_grad > 0.0)
                {
                    _dir[i] = -1.0*l_grad;
                }
                else
                {
                    _dir[i] = 0.0;
                }
            }
        }
        // check dir direction, same direction with -grad
        for (int i=0; i<_N; ++i)
        {
            if (_dir[i]*_grad[i] > 0.0)
            {
                _dir[i] = 0.0;
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
        _sy[i] = dot(_S[i], _Y[i]);
        _alpha[i] = dot(_S[i], _dir) / _sy[i];
        for (size_t k=0; k<_dir.size(); ++k)
        {
            _dir[k] -= _alpha[i]*_Y[i][k]
        }
    }
    // H0 = I*(s'y/y'y)
    i = (_end-1)%M;
    double yy = _sy[i]/dot(_Y[i], _Y[i]);
    for (size_t k=0; k<_dir.size(); ++k)
    {
        _dir[k] *= yy;
    }
    // second loop
    i = _start;
    while (i != _end)
    {
        _beta = dot(_Y[i], _dir) / _sy[i];
        for (size_t k=0; k<_dir.size(); ++k)
        {
            _dir[k] -= _S[i][k]*(_beta + _alpha[i]);
        }
        i = (i+1)%M;
    }
}

void OWLQN::linearSearch()
{
    // check is _dir down dir
    double res = 0.0;
}

void OWLQN::shift()
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

double OWLQN::dot(vector<double> x, vector<double> y)
{
    if (x.size() != y.size())
    {
        return 0.0;
    }
    double value = 0.0;
    for (size_t i=0; i<x.size(); ++i)
    {
        value += x[i]*y[i];
    }
    return value;
}