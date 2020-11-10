#pragma once

#include <array>
#include "utils.h"


template <size_t N>
double Evaluate(const std::array<double, N + 1>& coefficients, const std::vector<double>& row, const std::array<int, N>& columnIndices)
{
    double ret = coefficients[N];
    for (size_t i = 0; i < N; ++i)
        ret += coefficients[i] * row[columnIndices[i]];
    return ret;
}

template <size_t N>
double RSquared(const std::array<double, N + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    Average averageSales;
    for (const auto& row : data.data)
        averageSales.AddSample(row[valueIndex]);

    double numerator = 0.0f;
    double denominator = 0.0f;
    for (const auto& row : data.data)
    {
        double actual = row[valueIndex];

        double estimate = Evaluate(coefficients, row, columnIndices);

        numerator += sqr(actual - estimate);
        denominator += sqr(actual - averageSales.average);
    }

    return 1.0f - numerator / denominator;
}

template <size_t N>
double AdjustedRSquared(const std::array<double, N + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    int predictorCount = int(N);
    int numSamples = (int)data.data.size();

    double rsquared = RSquared(coefficients, data, columnIndices, valueIndex);
    
    double numerator = (1.0f - rsquared) * double(numSamples - 1);
    double denominator = double(numSamples - predictorCount - 1);

    return 1.0f - numerator / denominator;
}

template <size_t N>
double LossFunction(const std::array<double, N + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    Average MSE;

    for (const auto& row : data.data)
    {
        double estimate = Evaluate(coefficients, row, columnIndices);

        double actual = row[valueIndex];

        double error = estimate - actual;

        MSE.AddSample(error * error);
    }

    return MSE.average;
}

template <size_t N>
void CalculateGradient(std::array<double, N + 1>& gradient, const std::array<double, N + 1>& _coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    // Calculates a gradient via central differences
    for (size_t index = 0; index <= N; ++index)
    {
        std::array<double, N + 1> coefficients = _coefficients;

        coefficients[index] = _coefficients[index] - c_epsilon;
        double A = LossFunction(coefficients, data, columnIndices, valueIndex);

        coefficients[index] = _coefficients[index] + c_epsilon;
        double B = LossFunction(coefficients, data, columnIndices, valueIndex);

        gradient[index] = (B - A) / (2.0f * c_epsilon);
    }
}