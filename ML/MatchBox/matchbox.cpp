/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 20:32
#
# Filename: matchbox.cpp
#
# Description: 
#
=============================================================================*/
#include "matchbox.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include "Common/string_util.h"

namespace ML {

void MatchBox::init(int k, std::string prior_m, std::string prior_v, double beta, int max_fea) {
	_k = k;
	_beta = beta;
	_user.resize(_k);
	_item.resize(_k);
	_s.resize(_k);
	_t.resize(_k);
	_z.resize(_k);
	clear_state();
	_w.reserve(max_fea);
	for (int i=0; i<_k; ++i)
	{
		_user[i].reserve(max_fea);
		_item[i].reserve(max_fea);
	}
	_user_prior.resize(_k);
	_item_prior.resize(_k);
	std::vector<std::string> prior_m_str;
	std::vector<std::string> prior_v_str;
	Common::splitString(prior_m, prior_m_str, ',');
	Common::splitString(prior_v, prior_v_str, ',');
	_w_prior.m = atof(prior_m_str[0].c_str());
	_w_prior.v = atof(prior_v_str[0].c_str());
	for (int i=1; i<=_k; ++i)
	{
		_user_prior[i].m = atof(prior_m_str[1].c_str());
		_user_prior[i].v = atof(prior_v_str[1].c_str());
		_item_prior[i].m = atof(prior_m_str[2].c_str());
		_item_prior[i].v = atof(prior_v_str[2].c_str());
	}
	LOG_INFO("%s", "Init MatchBox Successful!");
}

MatchBox::Param MatchBox::divGauss(const Param& p1, const Param& p2) {
	Param r;
	r.v = p1.v/(p2.v-p1.v)*p2.v;
	r.m = p1.m*(r.v/p1.v) - p2.m*(r.v/p2.v);
	return r;
}
MatchBox::Param MatchBox::multGauss(const Param& p1, const Param& p2) {
	Param r;
	r.v = p1.v/(p2.v+p1.v)*p2.v;
	r.m = p1.m*(r.v/p1.v) + p2.m*(r.v/p2.v);
	return r;
}
MatchBox::Param MatchBox::addGauss(const Param& p1, const Param& p2) {
	Param r;
	r.v = p1.v + p2.v;
	r.m = p1.m + p2.m;
	return r;
}
MatchBox::Param MatchBox::decGauss(const Param& p1, const Param& p2) {
	Param r;
	r.v = p1.v + p2.v;
	r.m = p1.m - p2.m;
	return r;
}
MatchBox::Param MatchBox::truncatedGauss(const Param& p, double a, double b) {
	Param r;
	double norm_a = (a-p.m)/sqrt(p.v);
	double norm_b = (b-p.m)/sqrt(p.v);
	double prob_a = 0.0;
	double prob_b = 0.0;
	double cprob_a = 0.0;
	double cprob_b = 0.0;
	if (norm_b > 10.0)
	{
		prob_a = gauss_probability(norm_a);
		cprob_a = cumulative_probability(norm_a);
		prob_b = 0.0;
		cprob_b = 1.0;
	}
	else if (norm_a < -10.0)
	{
		prob_a = 0.0;
		cprob_a = 0.0;
		prob_b = gauss_probability(norm_b);
		cprob_b = cumulative_probability(norm_b);
	}
	else
	{
		prob_a = gauss_probability(norm_a);
		cprob_a = cumulative_probability(norm_a);
		prob_b = gauss_probability(norm_b);
		cprob_b = cumulative_probability(norm_b);
	}
	double diff_prob = prob_b - prob_a;
	double total_prob = cprob_b - cprob_a;
	double w_diff_prob = prob_a*norm_a - prob_b*norm_b;
	double v = diff_prob/total_prob;
	r.m = p.m + sqrt(p.v)* v;
	r.v = p.v*(1+w_diff_prob/total_prob - v*v);
	return r;
}

double MatchBox::cumulative_probability(double  t, double mean, double variance)
{
    double m = (t - mean);
    if (fabs(m) > 40*sqrt(variance) )
    {
        return m < 0 ? 0.0 : 1.0;
    }
    return 0.5*(1 + erf( m / sqrt(2*variance) ) );
}

double MatchBox::gauss_probability(double t, double mean, double variance)
{
    const double PI = 3.1415926;
    double m = (t - mean) / sqrt(variance);
    return exp(-m*m/2) / sqrt(2*PI*variance);
}

void MatchBox::train(const LongMatrixFeature* sample, double label) {
	clear_state();
	// down passing
	// compute (sk->*), (tk->*), (b->+)
	const LongMatrixFeature* start = sample;
	while (sample->index != (uint64_t)-1)
	{
		if (sample->type & 2)
		{
			Param p = get_w_param(sample->index);
			p.m *= sample->value;
			p.v  *= sample->value*sample->value;
			_b = addGauss(_b, p);
		}
		if ( (sample->type & 1) == 0)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_user_param(sample->index, i);
				p.m *= sample->value;
				p.v  *= sample->value*sample->value;
				_s[i] = addGauss(_s[i], p);
			}
		}
		else if ( (sample->type & 1) == 1)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_item_param(sample->index, i);
				p.m *= sample->value;
				p.v  *= sample->value*sample->value;
				_t[i] = addGauss(_t[i], p);
			}
		}
		else {}
		sample++;
	}
	_b = addGauss(_b, _bias);
	// compute (*->zk) = (zk->+)
	Param total_z;
	for (int i=0; i<_k; ++i)
	{
		_z[i].m = _s[i].m *_t[i].m ; 
		_z[i].v = (_s[i].v + square(_s[i].m) ) * ( _t[i].v + square(_t[i].m)  ) - square(_s[i].m*_t[i].m);
		total_z = addGauss(total_z, _z[i]);
	}
	// compute (r->+)
	_r = addGauss(_r, total_z);
	_r = addGauss(_r, _b);
	// compute (+->r')
	_r.v += _beta*_beta;
	LOG_DEBUG("total_z : [%lf, %lf]", total_z.m, total_z.v);
	LOG_DEBUG("_r : [%lf, %lf]", _r.m, _r.v);
	// compute p(r') = (+->r')*P(r'>0)
	Param r_post;
	if (label > 0.5)
	{
		r_post = truncatedGauss(_r, 0, 1000);
	}
	else
	{
		r_post = truncatedGauss(_r, -1000, 0);
	}
	
	LOG_DEBUG("r_post : [%lf, %lf]", r_post.m, r_post.v);
	// up passing
	// compute (r'->+) = p(r')/(+->r')
	_r = divGauss(r_post, _r);
	// compute (+->r) = (r->+)
	_r.v += _beta*_beta;
	// compute (+->b) = (r->+) - (Z->+)=(b->+)
	Param UP_b = decGauss(_r, total_z);
	// compute  (+->zk) = (zk->*)
	Param Up_z = decGauss(_r, _b);
	Param Up_zk = decGauss(Up_z, total_z);
	for (int i=0; i<_k; ++i)
	{
		_z[i].m = Up_zk.m + _z[i].m;
		_z[i].v = Up_zk.v - _z[i].v; 
		//  compute (*->sk) = (sk->+)
		double tk_second_moment = second_moment(_t[i]);
		Param Up_sk;
		Up_sk.m = _z[i].m*_t[i].m/tk_second_moment;
		Up_sk.v = _z[i].v / tk_second_moment;
		//  compute (*->tk) = (tk->+)
		double sk_second_moment = second_moment(_s[i]);
		Param Up_tk;
		Up_tk.m = _z[i].m*_s[i].m/sk_second_moment;
		Up_tk.v = _z[i].v / sk_second_moment;
		// compute (+->u_ki) - u_ki 
		_s[i] = decGauss(Up_sk, _s[i]);
		_t[i] = decGauss(Up_tk, _t[i]);
	}
	// (+->w) - w
	_b = decGauss(UP_b, _b);
	// compute w_post, user_post, item_post
	// compute bias_post = prior(bias) * (+->bias)
	Param Up_bias;
	Up_bias.m = _b.m + _bias.m;
	Up_bias.v = _b.v - _bias.v;
	_bias = multGauss(_bias,  Up_bias);
	sample = start;
	while (sample->index != (uint64_t)-1)
	{
		if (sample->type & 2)
		{
			Param p = get_w_param(sample->index);
			Param Up_w;
			Up_w.m = _b.m + p.m;
			Up_w.m /= sample->value;
			Up_w.v = _b.v - p.v;
			Up_w.v /= square(sample->value);
			p = multGauss(Up_w, p);
			set_w_param(sample->index, p);
		}
		if ( (sample->type & 1)== 0)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_user_param(sample->index, i);
				Param Up_u;
				Up_u.m = (_s[i].m + p.m)/sample->value;
				Up_u.v = (_s[i].v - p.v)/square(sample->value);
				p = multGauss(Up_u, p);
				set_user_param(sample->index, i, p);
			}
		}
		else if ( (sample->type & 1) == 1)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_item_param(sample->index, i);
				Param Up_v;
				Up_v.m = (_t[i].m + p.m)/sample->value;
				Up_v.v = (_t[i].v - p.v)/square(sample->value);
				p = multGauss(Up_v, p);
				set_item_param(sample->index, i, p);
			}
		}
		else {}
		sample++;
	}
}

double MatchBox::predict(const LongMatrixFeature* sample) {
	clear_state();
	// down passing
	// compute (sk->*), (tk->*), (b->+)
	while (sample->index != (uint64_t)-1)
	{
		if (sample->type & 2)
		{
			Param p = get_w_param(sample->index);
			p.m *= sample->value;
			p.v  *= sample->value*sample->value;
			_b = addGauss(_b, p);
		}
		if ( (sample->type & 1)== 0)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_user_param(sample->index, i);
				p.m *= sample->value;
				p.v  *= sample->value*sample->value;
				_s[i] = addGauss(_s[i], p);
			}
		}
		else if ( (sample->type & 1) == 1)
		{
			for (int i=0; i<_k; ++i)
			{
				Param p = get_item_param(sample->index, i);
				p.m *= sample->value;
				p.v  *= sample->value*sample->value;
				_t[i] = addGauss(_t[i], p);
			}
		}
		else {}
		sample++;
	}
	_b = addGauss(_b, _bias);
	// compute (*->zk) = (zk->+)
	Param total_z;
	for (int i=0; i<_k; ++i)
	{
		_z[i].m = _s[i].m *_t[i].m ; 
		_z[i].v = (_s[i].v + square(_s[i].m) ) * ( _t[i].v + square(_t[i].m)  ) - square(_s[i].m*_t[i].m);
		total_z = addGauss(total_z, _z[i]);
	}
	// compute (r->+)
	_r = addGauss(_r, total_z);
	_r = addGauss(_r, _b);
	// compute (+->r')
	_r.v += _beta*_beta;
	return cumulative_probability(_r.m/_r.v);
}

void MatchBox::saveMap(std::ostream& out_file, ParamMap& map)
{
	out_file << map.size() << std::endl;
	for (ParamMap::const_iterator iter = map.begin(); iter != map.end(); ++iter)
	{
	        out_file << iter->first << "\t" 
	            << iter->second.m << "\t"
	            << iter->second.v << std::endl;
	}
}

void MatchBox::loadMap(std::istream& infile, ParamMap& map)
{
	std::string line;
	getline(infile, line);
	int size = atoi(line.c_str());
	map.reserve(size);
	for (int i=0; i<size; ++i)
	{
		getline(infile, line);
		uint64_t fea_idx;
		Param tmp;
		sscanf(line.c_str(), "%lu\t%lf\t%lf", &fea_idx, &(tmp.m), &(tmp.v));
		map[fea_idx] = tmp;
	}
}

void MatchBox::save_model(const std::string& file) {
	std::ofstream out_file(file.c_str());
	out_file << _k << std::endl;
	out_file << _beta << std::endl;
    	// save prior param
    	out_file <<  _w_prior.m << "\t" << _w_prior.v << std::endl;
    	for (int i=0; i<_k; ++i)
    	{
    		out_file << _user_prior[i].m << "\t" << _user_prior[i].v << std::endl;
    	}
    	for (int i=0; i<_k; ++i)
    	{
    		out_file << _item_prior[i].m << "\t" << _item_prior[i].v << std::endl;
    	}
    	// save bias param
    	out_file << _bias.m << "\t"
            		<< _bias.v << std::endl;
            	saveMap(out_file, _w);
            	for (int i=0; i<_k; ++i)
            	{
            		saveMap(out_file, _user[i]);
            	}
            	for (int i=0; i<_k; ++i)
            	{
            		saveMap(out_file, _item[i]);
            	}
            	out_file.close();
}

void MatchBox::load_model(const std::string& file) {
	std::ifstream infile(file.c_str());
	std::string line;
	// param
	getline(infile, line);
	_k = atoi(line.c_str());
	_user_prior.resize(_k);
	_item_prior.resize(_k);
	_user.resize(_k);
	_item.resize(_k);
	_s.resize(_k);
	_t.resize(_k);
	_z.resize(_k);
	getline(infile, line);
	_beta = atof(line.c_str());
	getline(infile, line);  
	sscanf(line.c_str(), "%lf\t%lf", &_w_prior.m, &_w_prior.v);
	for (int i=0; i<_k; ++i)
    	{
    		getline(infile, line);
    		sscanf(line.c_str(), "%lf\t%lf", &(_user_prior[i].m), &(_user_prior[i].v));
    	}
    	for (int i=0; i<_k; ++i)
    	{
    		getline(infile, line);
    		sscanf(line.c_str(), "%lf\t%lf", &(_item_prior[i].m), &(_item_prior[i].v));
    	}
	getline(infile, line);  
	sscanf(line.c_str(), "%lf\t%lf", &(_bias.m), &(_bias.v));
	loadMap(infile, _w);
	for (int i=0; i<_k; ++i)
	{
		loadMap(infile, _user[i]);
	}
	for (int i=0; i<_k; ++i)
	{
		loadMap(infile, _item[i]);
	}
	infile.close();
}

MatchBox::Param MatchBox::get_user_param(uint64_t idx, int k) {
	if (_user[k].find(idx) != _user[k].end())
	{
		return _user[k][idx];
	}
	return _user_prior[k];
}
MatchBox::Param MatchBox::get_item_param(uint64_t idx, int k) {
	if (_item[k].find(idx) != _item[k].end())
	{
		return _item[k][idx];
	}
	return _item_prior[k];
}

MatchBox::Param MatchBox::get_w_param(uint64_t idx) {
	if (_w.find(idx) != _w.end())
	{
		return _w[idx];
	}
	return _w_prior;
}
void MatchBox::clear_state() {
	for (int i=0; i<_k; ++i)
	{
		_s[i].reset();
		_t[i].reset();
		_z[i].reset();
	}
	_b.reset();
	_r.reset();
}

}
