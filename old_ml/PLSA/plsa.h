/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-07 08:46
#
# Filename: plsa.h
#
# Description: plsa simple interface, dense matrix
#
=============================================================================*/
#ifndef _PLSA_H_
#define _PLSA_H_

#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include "ML/Common/log.h"
#include "ML/Common/string_util.h"
#include "ML/Common/double_buffer.h"

namespace ML {

// def Word, Doc
// typedef uint64_t WordType;
typedef std::string WordType;
struct Word
{
    WordType word;
    uint32_t count;
};
typedef std::vector<Word> Doc;
// def P(w|z)
typedef std::map<WordType, double> WordProbDict;
typedef std::vector<WordProbDict> TopicWordProb;
// def P(z|d)
typedef std::vector<double> TopicProbs;
typedef std::vector<TopicProbs > DocTopicProb;
DOUBLE_BUFFER(DocTopicProb)
DOUBLE_BUFFER(TopicWordProb)

class PLSA {
public:
    PLSA(){}
    ~PLSA(){}
    void init(uint32_t tn, uint32_t mi);
    void train();
    void load_data(std::string data_file);
    void save_model(std::string model_file);
    void load_model(std::string model_file);
private:
    void EM();
private:
    uint32_t _topic_num;
    uint32_t _max_iter;
private:
    std::vector<Doc> _doc_words; // C(d, w), don't store P(z|d, w) if using two buffer P(w|z) and P(z|d)
    DocTopicProb_Double_Buffer _doc_topic_probs; // P(z|d)
    TopicWordProb_Double_Buffer _topic_word_probs; // P(w|z)
};

}

#endif  // _PLSA_H_
