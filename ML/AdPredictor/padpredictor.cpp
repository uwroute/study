/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 15:07
#
# Filename: mt_adpredictor.cpp
#
# Description: mt_adpredictor
#
=============================================================================*/

#include "padpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include <time.h>
#include <unistd.h>

namespace ML {


void ParameterServer::update(const uint64_t& idx, const Message& msg)
{
	if (_messages.find(idx) == _messages.end())
	{
		_messages[idx] = msg;
	}
	else
	{
		_messages[idx].vMsg += msg.vMSg;
		_messages[idx].mMsg += msg.mMsg;
	}
}
Message ParameterServer::get(const uint64_t& idx)
{
	Message msg;
	if (_messages.find(idx) != _messages.end())
	{
		msg = _messages[idx];
	}
	return msg;
}
void ParameterServer::update(const UpdateRequest& req)
{
	Common::WLock wlock(_rw_mutex);
	for (size_t i=0; i<req.idxs.sie(); ++i)
	{
		update(req.idxs[i], req.msgs[i]);
	}
}

void ParameterServer::get(const GetRequest& req, GetResponse& res)
{
	Common::RLock rlock(_rw_mutex);
	res.idxs.reserve(req.idxs.size());
	res.msgs.reserve(req.idxs.size());
	for (size_t i=0; i<req.idxs.size(); ++i)
	{
		res.idxs.push_back(req.idxs[i]);
		res.msgs.push_back(get(req.idxs[i]));
	}
}

void ParameterServer::save_params(std::ofstream& out_file)
{
    double mean = 0.0, variance = 0.0;
    out_file << _messages.size() << std::endl;
    for (MessageHashMap::const_iterator iter = _messages.begin(); iter != _messages.end(); ++iter)
    {
        variance = 1.0/_messages[iter->first].vMsg;
        mean = variance*_messages[iter->first].mMsg;
        out_file << iter->first << "\t" 
            << mean << "\t"
            << variance << std::endl;
    }
}

void AdPredictorClient::init(double mean, double variance, double beta, double eps, int mini_batch, size_t max_fea_num, uint64_t _bias_idx, bool use_bias, double down_sample, int up_sample, bool update)
{
    // model param
    _prior_param.m = mean;
    _prior_param.v = variance;
    _use_bias = use_bias;
    _bias_param.m = _prior_param.m;
    _bias_param.v = _prior_param.v;
    _beta = beta;
    _eps = eps;
    // bias
    _bias_idx = bias_idx;
    _user_bias = use_bias;
    // ps train param
    _down_sample = down_sample;
    _up_sample = up_sample;
    _is_update = update;
    // param
    _messages.reserve(max_fea_num);
    _params.reserve(max_fea_num);
}

void AdPredictorClient::update_message(const uint64_t idx, const Message& param)
{
    Message new_msg = get_message(idx);
    new_msg.vMsg += param.vMsg;
    new_msg.mMsg += param.mMsg;
    _messages[idx] = new_msg;
}

Message AdPredictorClient::AdPredictorSlave::get_message(uint64_t idx)
{
    Message msg;
    if (_messages.find(idx) != _messages.end())
    {
        return _messages[idx];
    }
    return msg;
}

void AdPredictorClient::set_param(const uint64_t idx, const Param& param)
{
    _params[idx] = param;
}

Param AdPredictorClient::get_param(uint64_t idx)
{
    if (_params.find(idx) != _params.end())
    {
        return _params[idx];
    }
    return _prior_param;
}

void AdPredictorClient::train(LongFeature* sample, double label)
{
    if (label < 0.5)
    {
        label = -1.0;
    }
    double total_mean=0.0, total_variance=0.0;
    active_mean_variance(sample, total_mean, total_variance);
    LOG_INFO("total_mean : %lf, total_variance : %lf", total_mean, total_variance);
    double t = label*total_mean/sqrt(total_variance);
    if (fabs(t) > 5.0)
    {
        t = t < 0 ? -5.0 : 5.0;
    }
    double v = gauss_probability(t, 0.0, 1.0) / cumulative_probability(t, 0.0, 1.0);
    double w = v*(v + t);
    LOG_TRACE("v : %lf, w : %lf", v, w);
    while (sample->index !=  (uint64_t)-1)
    {
        Param cur_param = get_param(sample->index);
        double mean = cur_param.m;
        double variance = cur_param.v;

        mean += label*sample->value*variance/sqrt(total_variance)*v;
        variance *=  1 - sample->value*sample->value*variance/total_variance*w;

        double rectify_variance = _prior_param.v*variance/( (1-_eps)* _prior_param.v + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_prior_param.m/_prior_param.v );

        Message msg;
        msg.vMsg = 1.0/rectify_variance - 1.0/cur_param.v;
        msg.mMsg = rectify_mean/rectify_variance - cur_param.m/cur_param.v;
        update_message(sample->index, msg);

        if (_update)
        {
		cur_param.m = rectify_mean;
		cur_param.v = rectify_variance;
		set_param(sample->index, cur_param);
        }
        LOG_TRACE("fea_index : %lu,  mean : %lf, variance : %lf", sample->index, rectify_mean, rectify_variance);
        sample++;
    }
    if (_use_bias)
    {
        double mean = _bias_param.m;
        double variance = _bias_param.v;
        LOG_TRACE("Client before bias , mean : %lf,  variance : %lf", mean, variance);

        mean += label*_bias_val*variance/sqrt(total_variance)*v;
        variance *=  1 - fabs(_bias_val)*variance/total_variance*w;

        double rectify_variance = _prior_param.v*variance/( (1-_eps)*_prior_param.v + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_prior_param.m/_prior_param.v );

        _bias_message.vMsg += 1.0/rectify_variance - 1.0/_bias_param.v;
        _bias_message.mMsg += rectify_mean/rectify_variance - _bias_param.m/_bias_param.v;

        if (_update)
        {
	 _bias_param.m = rectify_mean;
	 _bias_param.v = rectify_variance;
        }
        
        LOG_TRACE("Client bias , mean : %lf,  variance : %lf", rectify_mean, rectify_variance);
    }
}

void AdPredictorClient::active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance)
{
    total_mean = 0.0;
    total_variance = 0.0;
    while (sample->index != (uint64_t)-1)
    {
        Param cur_param = get_param(sample->index);
        total_mean += cur_param.m * sample->value;
        total_variance += cur_param.v* sample->value*sample->value;
        sample++;
    }
    if (_use_bias)
    {
        total_mean += _bias_param.m * _bias_val;
        total_variance += _bias_param.v * _bias*_bias_val;
    }
    total_variance += _beta*_beta;
}

double AdPredictorClient::cumulative_probability(double  t, double mean, double variance)
{
    double m = (t - mean);
    if (fabs(m) > 40*sqrt(variance) )
    {
        return m < 0 ? 0.0 : 1.0;
    }
    return 0.5*(1 + erf( m / sqrt(2*variance) ) );
}

double AdPredictorClient::gauss_probability(double t, double mean, double variance)
{
    const double PI = 3.1415926;
    double m = (t - mean) / sqrt(variance);
    return exp(-m*m/2) / sqrt(2*PI*variance);
}

void AdPredictorClient::train_minibatch(LongDataSet& data)
{
	LOG_TRACE("Slave %d Train Mini Data", _seri);
	Request req;
	Response res;
	form_query_request(data, req);
	LOG_TRACE("Slave %d get message from master", _seri);
	_ps->get(req, res);
	update_param(res);
	LOG_TRACE("Slave %d has get message from master", _seri);
	req.clear();
	for (int i=0; i<data.sample_num; ++i)
	{
		LongFeature* sample = &(data.samples[data.sample_idx[i]]);
		if (data.labels[i] > 0.5 && _up_sample > 1)
		{
			for (int k=0; k<_up_sample; ++k)
			{
				train(sample, data.labels[i]);
			}
		}
		else if (data.labels[i] < 0.0)
		{
			if (rand()%RAND_MAX < _down_sample)
			{
				train(sample, data.labels[i]);
			}
		}
		else
		{
			train(sample, data.labels[i]);
		}
	}
	form_update_request(req);
	LOG_TRACE("Slave %d update message to master", _seri);
	_ps->update(req);
	LOG_TRACE("Slave %d has update message to master", _seri);
}

void AdPredictorClient::form_get_request(LongDataSet& data, GetRequest& req)
{
	_messages.clear();
	_bias_message.mMsg = 0.0;
	_bias_message.vMsg = 0.0;
	_params.clear();
	_bias_param = _prior_param;
	if (_use_bias)
	{
		req.messages_idx.push_back(_bias_idx);
	}
	for (int i=0; i<data.sample_num; ++i)
	{
		LongFeature* sample = &(data.samples[data.sample_idx[i]]);
		while (sample->index != (uint64_t)-1)
		{
			if (_params.find(sample->index) == _params.end())
			{
				_params[sample->index] = _prior_param;
				req.idxs.push_back(sample->index);
			}
			sample++;
		}
	}
}
void AdPredictorClient::form_update_request(UpdateRequest& req)
{
	if (_use_bias)
	{
		req.idxs.push_back(_bias_idx);
		req.msgs.push_back(_bias_message);
	}
	LOG_TRACE("bias msg : idx=%lu, mMsg=%lf, vMsg=%lf", req.messages_idx[0], req.messages_val[0].mMsg, req.messages_val[0].vMsg);
	for (MessageHashMap::iterator iter=_messages.begin(); iter!=_messages.end(); ++iter)
	{
		req.idxs.push_back(iter->first);
		req.msgs.push_back(iter->second);
		LOG_TRACE("msg : idx=%lu, mMsg=%lf, vMsg=%lf", iter->first, iter->second.mMsg, iter->second.vMsg);
	}
}
void AdPredictorSlave::update_param(GetResponse& res)
{
	for (size_t i=0; i<res.idxs.size(); ++i)
	{
		Param cur_param;
		cur_param.v = 1.0/res.messages_val[i].vMsg;
		cur_param.m = res.messages_val[i].mMsg*cur_param.v;
		if (res.idxs[i] == _bias_idx)
		{
			_bias_param = cur_param;
			LOG_TRACE("update bias : m=%lf, v=%lf", _bias_param.m, _bias_param.v);
		}
		else
		{
			set_param(res.messages_idx[i], cur_param);
		}
	}
}

}
