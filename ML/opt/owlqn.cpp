#include <owlqn.h>

void OWLQN::optimize()
{
    // compute dir
    // compute grad
    grad();
    dir();
    linearSearch();
    shift();
}

void OWLQN::grad()
{
    model->grad(_grad);
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
                _dir[i] = -1.0*(_dir[i] + _l1)
            }
            else if (_w[i] < -1.0*MinDoubleValue)
            {
                _dir[i] = -1.0*(_dir[i] - _l1)
            }
            else
            {
                double l_grad = _dir[i]-_l1;
                double r_grad = _dir[i]+_l1;
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
    }
    // two loop lbfgs

}

void OWLQN::linearSearch()
{
}

void OWLQN::shift()
{

}
