#include "data.h"
#include <iostream>

int main(int argc, char* argv[])
{
    char* infile = argv[1];
    ML::DataSet data;
    ML::load_data(infile, data);
    return 0;
}
