#pragma once

#include <array>
#include "utils.h"

template <size_t N>
float Evaluate(const std::array<float, N * 2 + 1>& coefficients, const std::vector<float>& row, const std::array<int, N>& columnIndices)
{
    float ret = coefficients[N * 2];
    for (size_t i = 0; i < N; ++i)
    {
        float x = row[columnIndices[i]];
        ret += coefficients[i * 2 + 0] * x * x;
        ret += coefficients[i * 2 + 1] * x;
    }
    return ret;
}

template <size_t N>
float RSquared(const std::array<float, N * 2 + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    Average averageSales;
    for (const auto& row : data.data)
        averageSales.AddSample(row[valueIndex]);

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (const auto& row : data.data)
    {
        float actual = row[valueIndex];

        float estimate = Evaluate(coefficients, row, columnIndices);

        numerator += sqr(actual - estimate);
        denominator += sqr(actual - averageSales.average);
    }

    return 1.0f - numerator / denominator;
}

template <size_t N>
float AdjustedRSquared(const std::array<float, N * 2 + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    int predictorCount = int(N);
    int numSamples = (int)data.data.size();

    float rsquared = RSquared(coefficients, data, columnIndices, valueIndex);
    
    float numerator = (1.0f - rsquared) * float(numSamples - 1);
    float denominator = float(numSamples - predictorCount - 1);

    return 1.0f - numerator / denominator;
}

template <size_t N>
float LossFunction(const std::array<float, N * 2 + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    Average MSE;

    for (const auto& row : data.data)
    {
        float estimate = Evaluate(coefficients, row, columnIndices);

        float actual = row[valueIndex];

        float error = estimate - actual;

        MSE.AddSample(error * error);
    }

    return MSE.average;
}

template <size_t N>
void CalculateGradient(std::array<float, N * 2 + 1>& gradient, const std::array<float, N * 2 + 1>& _coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    // Calculates a gradient via central differences
    for (size_t index = 0; index <= N * 2; ++index)
    {
        std::array<float, N * 2 + 1> coefficients = _coefficients;

        coefficients[index] = _coefficients[index] - c_epsilon;
        float A = LossFunction(coefficients, data, columnIndices, valueIndex);

        coefficients[index] = _coefficients[index] + c_epsilon;
        float B = LossFunction(coefficients, data, columnIndices, valueIndex);

        gradient[index] = (B - A) / (2.0f * c_epsilon);
    }
}