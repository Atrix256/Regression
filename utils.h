#pragma once

#include <vector>
#include <string>
#include <unordered_map>

inline float Lerp(float A, float B, float t)
{
    return A * (1.0f - t) + B * t;
}

struct CSV
{
    std::vector<std::string> headers;
    std::vector<std::vector<float>> data;

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
    float average = 0.0f;

    void AddSample(float sample)
    {
        samples++;
        average = Lerp(average, sample, 1.0f / float(samples));
    }
};

bool LoadCSV(const char* fileName, CSV& csv);

void Model1(const CSV& train, const CSV& test);
void Model2(const CSV& train, const CSV& test);
void Model3(const CSV& train, const CSV& test);
