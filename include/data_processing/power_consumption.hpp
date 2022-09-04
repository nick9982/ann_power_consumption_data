#include <stdlib.h>
#include <vector>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <random>

using namespace std;

class minMaxFinder
{
    public:
        minMaxFinder();

        void isMinOrMax(double x);

        double min;
        double max;
};

class dataset
{
    public:
        vector<vector<string>> unnormalized_data;
        vector<vector<double>> data;
        vector<minMaxFinder> unnorminfo;
        uint size;
        string filepath;
        string name;
        vector<string> column_labels;

        dataset(uint size, string filepath, string name);

        void readFile();

        void normalize();

        double normalizeTime(string time);

        double minMaxNormalization(double x, minMaxFinder mmf);

        double minMaxUnnormalization(double x, uint column);

        vector<dataset> split(uint splitAt, string name1, string name2);

        void setData(uint i, uint j, double val);

        void peak(uint limit);

        string column_labels_toString();

        void shuffle();
};