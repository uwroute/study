/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-21 09:40
#
# Filename: dtree.h
#
# Description: 
#
=============================================================================*/
#ifndef _DTREE_H_
#define _DTREE_H_

enum FeaType
{
    MUTICLASS = 0,
    CONTINUES = 1,
};

struct TreeNode
{
    uint64_t _index;
    FeaType _fea_type;
    double _split_value;
    std::map<int, size_t> _child;
    double _value;
};

class Tree {
public:
    Tree() {}
    ~Tree() {}
    void buildTree();
    void saveModel();
    void loadModel();
    double predict();
private:
    std::vector<TreeNode> _tree;
};

#endif  // _DTREE_H_
