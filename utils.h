#pragma once

#include <vector>
#include <string>
#include <unordered_map>

inline double Lerp(double A, double B, double t)
{
    return A * (1.0f - t) + B * t;
}

inline double sqr(double x)
{
    return x * x;
}

struct CSV
{
    std::vector<std::string> headers;
    std::vector<std::vector<double>> data;

    int GetHeaderIndex(const char* h) const
    {
        int index = -1;
        for (const std::string& header : headers)
        {
            index++;
            if (header == h)
                return index;
        }
        return -1;
    }
};

struct Average
{
    int samples = 0;
    double average = 0.0f;

    void AddSample(double sample)
    {
        samples++;
        average = Lerp(average, sample, 1.0f / double(samples));
    }
};

bool LoadCSV(const char* fileName, CSV& csv);

void Model1(const CSV& train, const CSV& test);
void Model2(const CSV& train, const CSV& test);
void Model3(const CSV& train, const CSV& test);
void Model4(const CSV& train, const CSV& test);
void Model5(const CSV& train, const CSV& test);
void Model6(const CSV& train, const CSV& test);
void Model7(const CSV& train, const CSV& test);
