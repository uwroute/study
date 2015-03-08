/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-07 08:48
#
# Filename: plsa.cpp
#
# Description: plsa simple interface, using dense matrix
#
=============================================================================*/

#include "plsa.h"
#include <fstream>
#include <stdlib.h>
#include <ctime>

namespace ML
{
using namespace Common;

#define SAFETY_INSERT(word, p_zw) \
for (size_t i=0; i<_topic_num; ++i) \
{                                   \
    if (p_zw[i].find(word) == p_zw[i].end()) \
    {                               \
        /*LOG_TRACE("insert %lu into P(w|topic_%lu)", word, i);*/ \
        p_zw[i][word] = 0.0;        \
    }                               \
}   

void PLSA::init(uint32_t tn, uint32_t mi)
{
    _topic_num = tn;
    _max_iter = mi;
    TopicWordProb& cur_p_zw = _topic_word_probs.cur();
    TopicWordProb& next_p_zw = _topic_word_probs.next();
    cur_p_zw.resize(_topic_num);
    next_p_zw.resize(_topic_num);
}

// EM train
void PLSA::EM()
{
    // get cur and next P(z|d),P(w|z)
    DocTopicProb& cur_p_dz = _doc_topic_probs.cur();
    DocTopicProb& next_p_dz = _doc_topic_probs.next();
    TopicWordProb& cur_p_zw = _topic_word_probs.cur();
    TopicWordProb& next_p_zw = _topic_word_probs.next();
    
    // 迭代初始化 next P(w|z)
    for (size_t i=0; i<next_p_zw.size(); ++i)
    {
        for (WordProbDict::iterator iter=next_p_zw[i].begin();
                iter != next_p_zw[i].end(); ++iter)
        {
            iter->second = 0.0;
        }
    }

    // P(z'|d, w) = P(z|d)*P(w|z)/sum(P(z|d)*P(w|z))
    size_t docs = _doc_words.size();
    // 遍历文档每一个单词
    for (size_t d=0; d<docs; ++d)
    {
        LOG_TRACE("Process %lu doc", d);
        uint32_t doc_word_count = 0;
        // 迭代初始化 next P(z|d)
        for (size_t i=0; i<_topic_num; ++i)
        {
            next_p_dz[d][i] = 0.0;
        }
        for (Doc::const_iterator iter=_doc_words[d].begin(); iter!=_doc_words[d].end(); ++iter)
        {
            // E step
            // P(z|d,w)
            std::vector<double> topic_prob(_topic_num, 0.0);
            // 一个d，w组合
            WordType w = iter->word;
            uint32_t word_count = iter->count;
            double sum = 0.0;
            for (size_t i=0; i<_topic_num; ++i)
            {
                topic_prob[i] = cur_p_zw[i][w] * cur_p_dz[d][i];
                sum += topic_prob[i];
            }
            for (size_t i=0; i<_topic_num; ++i)
            {
                topic_prob[i] /= sum;
                // M step
                // count*P(z|d,w)
                double topic_w = word_count*topic_prob[i];
                // next P(z|d) P(w|z)
                next_p_dz[d][i] += topic_w;
                next_p_zw[i][w] += topic_w;
            }
            doc_word_count += word_count;
        }
        for (size_t i=0; i<_topic_num; ++i)
        {
            next_p_dz[d][i] /= doc_word_count;
        }
    }
    for (size_t i = 0; i<next_p_zw.size(); ++i)
    {
        double total = 0.0;
        for (WordProbDict::const_iterator iter = next_p_zw[i].begin(); iter != next_p_zw[i].end(); ++iter)
        {
            total += iter->second;
        }
        for (WordProbDict::iterator iter = next_p_zw[i].begin(); iter != next_p_zw[i].end(); ++iter)
        {
            iter->second /= total;
        }
    }
    _doc_topic_probs.swap();
    _topic_word_probs.swap();
}

void PLSA::train()
{
    for (size_t iter = 0; iter < _max_iter; ++iter)
    {
        LOG_INFO("-------Iter %lu begin!--------", iter);
        EM();
        LOG_INFO("-------Iter %lu end!--------", iter);
    }
}

void PLSA::load_data(std::string data_file)
{
    std::ifstream infile(data_file.c_str());
    std::string line;
    getline(infile, line);
    TopicProbs topic_prob(_topic_num, 0.0);
    // get cur and next P(z|d),P(w|z)
    DocTopicProb& cur_p_dz = _doc_topic_probs.cur();
    DocTopicProb& next_p_dz = _doc_topic_probs.next();
    TopicWordProb& cur_p_zw = _topic_word_probs.cur();
    TopicWordProb& next_p_zw = _topic_word_probs.next();
    srand(time(NULL));
    // load a doc
    while (!infile.eof())
    {
        std::vector<Feature> sample;
        double label;
        if (0 == toSample(line, sample, label))
        {
            getline(infile, line);
            continue;
        }
        Doc d;
        for (size_t i=0; i<sample.size(); ++i)
        {
            Word w;
            w.word = sample[i].index;
            w.count = sample[i].value;
            d.push_back(w);
            SAFETY_INSERT(w.word, cur_p_zw)
            SAFETY_INSERT(w.word, next_p_zw)
        }
        _doc_words.push_back(d);
        double max = 1.0;
        for (size_t i=0; i<_topic_num; ++i)
        {
            topic_prob[i] = max*rand()/RAND_MAX;
            max -= topic_prob[i];
        }
        cur_p_dz.push_back(topic_prob);
        next_p_dz.push_back(topic_prob);
        getline(infile, line);
    }
    // init cur_p_zw
    for (size_t i=0; i<cur_p_zw.size(); ++i)
    {
        double max = 1.0;
        for (WordProbDict::iterator iter = cur_p_zw[i].begin();
                iter != cur_p_zw[i].end(); ++iter)
        {
            iter->second = max*rand()/RAND_MAX;
            max -= iter->second;
        }
    }
    LOG_INFO("Topic Num : %lu", cur_p_dz.size());
    LOG_INFO("Doc Num : %lu", cur_p_zw.size());
    LOG_INFO("P(z|d) Size : %lu * %lu", cur_p_dz.size(), cur_p_dz[0].size());
    LOG_INFO("P(w|z) Size : %lu * %lu", cur_p_zw.size(), cur_p_zw[0].size());
    LOG_INFO("%s", "Load data successful!");
}

void PLSA::save_model(std::string model_file)
{
    std::ofstream out(model_file.c_str());
    TopicWordProb& p_zw = _topic_word_probs.cur();
    DocTopicProb& p_dz = _doc_topic_probs.cur();
    // save P(w|z)
    out << "P_ZW" << std::endl;
    for (size_t k=0; k<_topic_num; ++k)
    {
        out << "Topic " << k << std::endl;
        for (WordProbDict::const_iterator iter=p_zw[k].begin();
                iter != p_zw[k].end(); ++iter)
        {
            out << iter->first << "\t" << iter->second << std::endl;
        }
    }
    // save P(z|d)
    out << "P_DZ" << std::endl;
    for (size_t d=0; d<p_dz.size(); ++d)
    {
        out << d << "\t";
        for (size_t k=0; k<_topic_num; ++k)
        {
            out << p_dz[d][k] << " ";
        }
        out << std::endl;
    }
}

void PLSA::load_model(std::string model_file)
{
}

}
