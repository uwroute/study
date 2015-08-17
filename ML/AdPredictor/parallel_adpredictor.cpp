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

#include "parallel_adpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"
#include <time.h>
#include <unistd.h>

namespace ML {

void AdPredictorMaster::init(double mean, double variance, double beta, double eps, size_t max_fea_num, bool use_bias)
{
    _prior_message.vMsg = 1.0/variance;
    _prior_message.mMsg = mean/variance;
    _use_bias = use_bias;
    _bias_message.vMsg = _prior_message.vMsg;
    _bias_message.mMsg = _prior_message.mMsg;
    _beta = beta;
    _eps = eps;
    _messages.reserve(max_fea_num);
}

void AdPredictorMaster::SetRcvQueue(Common::MessageQueue<Request>* queue)
{
	_rcv_queue = queue;
}

void AdPredictorMaster::AddSlave(Common::MessageQueue<Response>* slave)
{
	_slave_queues.push_back(slave);
	_slave_status.push_back(false);
}

void AdPredictorMaster::update_message(const uint64_t idx, const Message& param)
{
	Message res = get_message(idx);
	res.mMsg += param.mMsg;
	res.vMsg += param.vMsg;
	_messages[idx] = res;
}
Message AdPredictorMaster::get_message(uint64_t idx)
{
	if (_messages.find(idx) != _messages.end())
	{
		return _messages[idx];
	}
	return _prior_message;
}

void AdPredictorMaster::update_bias_message(const Message& param)
{
	_bias_message.mMsg += param.mMsg;
	_bias_message.vMsg += param.vMsg;
	double variance = 1.0/_bias_message.vMsg;
	double mean = variance*_bias_message.mMsg;
	LOG_INFO("master bias , mean : %lf,  variance : %lf", mean,  variance);
}

void AdPredictorMaster::update_message(const Request& req)
{
	Common::WLock wlock(_rw_mutex);
	update_bias_message(req.messages_val[0]);
	for (size_t i=1; i<req.messages_idx.size(); ++i)
	{
		update_message(req.messages_idx[i], req.messages_val[i]);
	}
}

void AdPredictorMaster::get_message(const Request& req, Response& res)
{
	Common::RLock rlock(_rw_mutex);
	res.messages_idx.push_back(-1);
	res.messages_val.push_back(_bias_message);
	for (size_t i=0; i<req.messages_idx.size(); ++i)
	{
		if (_messages.find(req.messages_idx[i]) != _messages.end())
		{
			res.messages_idx.push_back(req.messages_idx[i]);
			res.messages_val.push_back(_messages[req.messages_idx[i]]);
		}
	}
}

void AdPredictorMaster::run()
{
	while (1)
	{
		if (_rcv_queue->size() > 0)
		{
			Request msg = _rcv_queue->pop();
			Response res;
			switch (msg.type)
			{
				case QUERY_PARAM:
					get_message(msg, res);
					break;
				case UPDATE_PARAM:
					update_message(msg);
					break;
				case END_TRAIN:
					//slave_staus = true;
					break;
				default:
					break;
			}
		}
		else
		{
			usleep(1);
		}
	}
}

void AdPredictorMaster::save_model(const std::string& file)
{
    std::ofstream out_file(file.c_str());
    double mean = 0.0, variance = 0.0;
    variance = 1.0/_prior_message.vMsg;
    mean = variance*_prior_message.mMsg;
    out_file << mean << std::endl;
    out_file << variance<< std::endl;
    out_file << _beta << std::endl;
    out_file << _eps << std::endl;
    out_file << _use_bias << std::endl;
    variance = 1.0/_bias_message.vMsg;
    mean = variance*_bias_message.mMsg;
    out_file << _bias << "\t"
            << mean << "\t"
            << variance << std::endl;
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

int AdPredictorMaster::load_model(const std::string& file)
{
    std::ifstream infile(file.c_str());
    std::string line;
    // param
    getline(infile, line);
    double mean = atof(line.c_str());
    getline(infile, line);
    double variance = atof(line.c_str());
    _prior_message.vMsg = 1.0/variance;
    _prior_message.mMsg = mean/variance;
    getline(infile, line);
    _beta = atof(line.c_str());
    getline(infile, line);
    _eps = atof(line.c_str());
    getline(infile, line); 
    _use_bias = atoi(line.c_str());
    getline(infile, line); 
    if (3 != sscanf(line.c_str(), "%lf\t%lf\t%lf", &_bias, &mean, &variance))
    {
        LOG_ERROR("Parser Bias Error :  %s", line.c_str());
        infile.close();
        return -1;
    }
    _bias_message.vMsg = 1.0/variance;
    _bias_message.mMsg = mean/variance;
    getline(infile, line); // get fea size
    int w_size = atoi(line.c_str());
    _messages.reserve(w_size);
    getline(infile, line);
    while (!infile.eof())
    {
        uint64_t fea = 0;
        if (3 != sscanf(line.c_str(), "%lu\t%lf\t%lf", &fea, &mean, &variance) )
        {
            LOG_ERROR("Parser Weight Error :  %s", line.c_str());
            infile.close();
            return -1;
        }
        Message msg;
        msg.vMsg = 1.0/variance;
        msg.mMsg = mean/variance;
        _messages[fea] = msg;
        getline(infile, line);
    }
    infile.close();
    LOG_INFO("Load Mode Size : %lu", _messages.size());
    return 0;
}

void AdPredictorSlave::init(double mean, double variance, double beta, double eps, int mini_batch, size_t max_fea_num, bool use_bias, double down_sample, bool update)
{
    _prior_param.m = mean;
    _prior_param.v = variance;
    _use_bias = use_bias;
    _bias_param.m = _prior_param.m;
    _bias_param.v = _prior_param.v;
    _beta = beta;
    _eps = eps;
    _down_sample = down_sample;
    _update = update;
    _mini_batch = mini_batch;
    _messages.reserve(max_fea_num);
    _params.reserve(max_fea_num);
}

void AdPredictorSlave::train(LongFeature* sample, double label)
{
    if (label < 0.5)
    {
        label = -1.0;
    }
    double total_mean=0.0, total_variance=0.0;
    active_mean_variance(sample, total_mean, total_variance);
    LOG_TRACE("total_mean : %lf, total_variance : %lf", total_mean, total_variance);
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
        LOG_INFO("Slave before bias , mean : %lf,  variance : %lf", mean, variance);

        mean += label*_bias*variance/sqrt(total_variance)*v;
        variance *=  1 - fabs(_bias)*variance/total_variance*w;

        double rectify_variance = _prior_param.v*variance/( (1-_eps)*_prior_param.v + _eps*variance );
        double rectify_mean = rectify_variance*( (1-_eps)*mean/variance + _eps*_prior_param.m/_prior_param.v );

        _bias_message.vMsg += 1.0/rectify_variance - 1.0/_bias_param.v;
        _bias_message.mMsg += rectify_mean/rectify_variance - _bias_param.m/_bias_param.v;

        if (_update)
        {
	 _bias_param.m = rectify_mean;
	 _bias_param.v = rectify_variance;
        }
        
        LOG_INFO("Slave bias , mean : %lf,  variance : %lf", rectify_mean, rectify_variance);
    }
}

void AdPredictorSlave::active_mean_variance(const LongFeature* sample, double& total_mean, double& total_variance)
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
        total_mean += _bias_param.m * _bias;
        total_variance += _bias_param.v * _bias*_bias;
    }
    total_variance += _beta*_beta;
}

void AdPredictorSlave::update_message(const uint64_t idx, const Message& param)
{
	Message new_param = get_message(idx);
	new_param.vMsg += param.vMsg;
	new_param.mMsg += param.mMsg;
	_messages[idx] = new_param;
}
Message AdPredictorSlave::AdPredictorSlave::get_message(uint64_t idx)
{
	Message msg;
	if (_messages.find(idx) != _messages.end())
	{
		return _messages[idx];
	}
	return msg;
}
void AdPredictorSlave::set_param(const uint64_t idx, const Param& param)
{
	_params[idx] = param;
}
Param AdPredictorSlave::get_param(uint64_t idx)
{
	if (_params.find(idx) != _params.end())
	{
		return _params[idx];
	}
	return _prior_param;
}


double AdPredictorSlave::cumulative_probability(double  t, double mean, double variance)
{
    double m = (t - mean);
    if (fabs(m) > 40*sqrt(variance) )
    {
        return m < 0 ? 0.0 : 1.0;
    }
    return 0.5*(1 + erf( m / sqrt(2*variance) ) );
}

double AdPredictorSlave::gauss_probability(double t, double mean, double variance)
{
    const double PI = 3.1415926;
    double m = (t - mean) / sqrt(variance);
    return exp(-m*m/2) / sqrt(2*PI*variance);
}

void AdPredictorSlave::train(std::string& file)
{
	std::ifstream infile(file);
	if (!infile)
    	{
    		LOG_ERROR("Open data file : %s failed!", file.c_str());
    		return;
    	}
    	LongDataSet data;
    	data.samples.reserve(_mini_batch*100);
    	data.sample_idx.reserve(_mini_batch);
    	data.labels.reserve(_mini_batch);
    	while (!infile.eof())
    	{
    		data.samples.clear();
    		data.sample_idx.clear();
    		data.labels.clear();
    		data.sample_num = 0;
    		data.sample_fea_num = 0;
    		load_data(infile, data, _mini_batch, 1.0);
    		train_minibatch(data);
    	}
    	infile.close();
}

void AdPredictorSlave::train_minibatch(LongDataSet& data)
{
	LOG_TRACE("Slave %d Train Mini Data", _seri);
	Request req;
	Response res;
	form_query_request(data, req);
	LOG_TRACE("Slave %d get message from master", _seri);
	_p_master->get_message(req, res);
	update_param(res);
	LOG_TRACE("Slave %d has get message from master", _seri);
	req.clear();
	for (int i=0; i<data.sample_num; ++i)
	{
		LongFeature* sample = &(data.samples[data.sample_idx[i]]);
		train(sample, data.labels[data.sample_idx[i]]);
	}
	form_update_request(req);
	LOG_TRACE("Slave %d update message to master", _seri);
	_p_master->update_message(req);
	LOG_TRACE("Slave %d has update message to master", _seri);
}

void AdPredictorSlave::form_query_request(LongDataSet& data, Request& req)
{
	_messages.clear();
	_bias_message.mMsg = 0.0;
	_bias_message.vMsg = 0.0;
	_params.clear();
	_bias_param = _prior_param;
	req.messages_idx.push_back(-1);
	req.type = UPDATE_PARAM;
	for (int i=0; i<data.sample_num; ++i)
	{
		LongFeature* sample = &(data.samples[data.sample_idx[i]]);
		while (sample->index != (uint64_t)-1)
		{
			if (_params.find(sample->index) == _params.end())
			{
				_params[sample->index] = _prior_param;
				req.messages_idx.push_back(sample->index);
			}
			sample++;
		}
	}
}
void AdPredictorSlave::form_update_request(Request& req)
{
	req.type = UPDATE_PARAM;
	req.messages_idx.push_back(-1);
	req.messages_val.push_back(_bias_message);
	LOG_INFO("bias msg : idx=%lu, mMsg=%lf, vMsg=%lf", req.messages_idx[0], req.messages_val[0].mMsg, req.messages_val[0].vMsg);
	for (MessageHashMap::iterator iter=_messages.begin(); iter!=_messages.end(); ++iter)
	{
		req.messages_idx.push_back(iter->first);
		req.messages_val.push_back(iter->second);
		LOG_TRACE("msg : idx=%lu, mMsg=%lf, vMsg=%lf", iter->first, iter->second.mMsg, iter->second.vMsg);
	}
}
void AdPredictorSlave::update_param(Response& res)
{
	_bias_param.v = 1.0/res.messages_val[0].vMsg;
	_bias_param.m = _bias_param.v*res.messages_val[0].mMsg;
	LOG_INFO("update bias : m=%lf, v=%lf", _bias_param.m, _bias_param.v);
	for (size_t i=1; i<res.messages_idx.size(); ++i)
	{
		Param cur_param;
		cur_param.v = 1.0/res.messages_val[i].vMsg;
		cur_param.m = res.messages_val[i].mMsg*cur_param.v;
		set_param(res.messages_idx[i], cur_param);
	}
}

}