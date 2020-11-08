#include "utils.h"
#include <array>
#include <random>

/*

Model 3:

f(x,y) = Ax + By + C

x and y are establishment year and MRP (price)
A, B and C are coefficients that were learned through gradient descent.

*/

// This is for central differences, for calculating the gradient
static const float c_epsilon = 0.01f;

// The learning rate, for gradient descent
static const float c_learningRate = 0.001f;

// how many steps of gradient descent are done
static const size_t c_gradientDescentSteps = 100;

// how many times should it pick a random set of parameters and do gradient descent?
static const size_t c_population = 100;


template <size_t N>
float LossFunction(const std::array<float, N + 1>& coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    // Our loss function is RMSE
    Average MSE;

    for (const auto& row : data.data)
    {
        float estimate = coefficients[N];
        for (size_t index = 0; index < N; ++index)
            estimate += row[columnIndices[index]] * coefficients[index];

        float actual = row[valueIndex];

        float error = estimate - actual;

        MSE.AddSample(error * error);
    }

    return sqrtf(MSE.average);
}

template <size_t N>
void CalculateGradient(std::array<float, N + 1>& gradient, const std::array<float, N + 1>& _coefficients, const CSV& data, const std::array<int, N>& columnIndices, int valueIndex)
{
    // Calculates a gradient via central differences
    for (size_t index = 0; index <= N; ++index)
    {
        std::array<float, N + 1> coefficients = _coefficients;

        coefficients[index] = _coefficients[index] - c_epsilon;
        float A = LossFunction(coefficients, data, columnIndices, valueIndex);

        coefficients[index] = _coefficients[index] + c_epsilon;
        float B = LossFunction(coefficients, data, columnIndices, valueIndex);

        gradient[index] = (B - A) / (2.0f * c_epsilon);
    }
}

void Model3(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - Linear fit of Item_Outlet_Sales based on Outlet_Establishment_Year and Item_MRP\n");

    // get the columns of interest
    int salesIndex = train.GetHeaderIndex("Item_Outlet_Sales");
    if (salesIndex == -1 || test.GetHeaderIndex("Item_Outlet_Sales") != salesIndex)
    {
        printf("Couldn't find Item_Outlet_Sales column.\n");
        return;
    }

    int yearIndex = train.GetHeaderIndex("Outlet_Establishment_Year");
    if (yearIndex == -1 || test.GetHeaderIndex("Outlet_Establishment_Year") != yearIndex)
    {
        printf("Couldn't find Outlet_Establishment_Year column.\n");
        return;
    }

    int MRPIndex = train.GetHeaderIndex("Item_MRP");
    if (MRPIndex == -1 || test.GetHeaderIndex("Item_MRP") != MRPIndex)
    {
        printf("Couldn't find Item_MRP column.\n");
        return;
    }

    // do gradient descent

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::array<int, 2> columnIndices = { yearIndex, MRPIndex };

    float bestLoss = FLT_MAX;
    std::array<float, 3> bestCoefficients;
    size_t bestCoefficientsPopulationIndex = 0;
    size_t bestCoefficientsStepIndex = 0;

    // for each member of the population
    for (size_t populationIndex = 0; populationIndex < c_population; ++populationIndex)
    {
        // random initialize some starting coefficients
        // NOTE: this does the same random numbers every time, so is deterministic, as written.
        std::array<float, 3> coefficients;
        for (float& f : coefficients)
            f = dist(rng);

        // keep the best coefficients seen
        float loss = LossFunction(coefficients, train, columnIndices, salesIndex);
        if (loss < bestLoss)
        {
            bestLoss = loss;
            bestCoefficients = coefficients;
            bestCoefficientsPopulationIndex = populationIndex;
            bestCoefficientsStepIndex = 0;
        }

        // do multiple steps of gradient descent
        for (int i = 0; i < c_gradientDescentSteps; ++i)
        {
            // calculate the gradient
            std::array<float, 3> gradient;
            CalculateGradient(gradient, coefficients, train, columnIndices, salesIndex);

            // descend
            for (size_t index = 0; index < 3; ++index)
                coefficients[index] -= gradient[index] * c_learningRate;

            // keep the best coefficients seen
            float loss = LossFunction(coefficients, train, columnIndices, salesIndex);
            if (loss < bestLoss)
            {
                bestLoss = loss;
                bestCoefficients = coefficients;
                bestCoefficientsPopulationIndex = populationIndex;
                bestCoefficientsStepIndex = i + 1;
            }
        }
    }

    // calculate mean squared error (average squared error) and root mean squared error from training data
    float Train_RMSE = LossFunction(bestCoefficients, train, columnIndices, salesIndex);
    float Test_RMSE = LossFunction(bestCoefficients, test, columnIndices, salesIndex);

    // Report results
    printf("  Best coefficients = %0.4f, %0.4f, %0.4f  (from %zu:%zu)\n", bestCoefficients[0], bestCoefficients[1], bestCoefficients[2], bestCoefficientsPopulationIndex, bestCoefficientsStepIndex);
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
