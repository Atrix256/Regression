// This is for central differences, for calculating the gradient
static const float c_epsilon = 0.01f;

// The learning rate, for gradient descent
static const float c_learningRate = 0.001f;

// how many steps of gradient descent are done
static const size_t c_gradientDescentSteps = 100;

// how many times should it pick a random set of parameters and do gradient descent?
static const size_t c_population = 100;

#include "utils.h"
#include "linearfit.h"
#include <random>

/*

Model 5:

f(x,y) = Ax + By + ... + Z

x, y, ... are the data columns
A, B, ..., Z are coefficients that were learned through gradient descent.

*/


void Model5(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - Linear fit of Item_Outlet_Sales based on all data items\n");

    // get the columns of interest
    int salesIndex = train.GetHeaderIndex("Item_Outlet_Sales");
    if (salesIndex == -1 || test.GetHeaderIndex("Item_Outlet_Sales") != salesIndex)
    {
        printf("Couldn't find Item_Outlet_Sales column.\n");
        return;
    }

    // do gradient descent
    // NOTE: this does the same random numbers every program run, so is deterministic, as written.
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::array<int, 35> columnIndices;

    for (int index = 0; index < 35; ++index)
    {
        if (index < salesIndex)
            columnIndices[index] = index;
        else
            columnIndices[index] = index + 1;
    }

    float bestLoss = FLT_MAX;
    std::array<float, 36> bestCoefficients;
    size_t bestCoefficientsPopulationIndex = 0;
    size_t bestCoefficientsStepIndex = 0;

    // for each member of the population
    for (size_t populationIndex = 0; populationIndex < c_population; ++populationIndex)
    {
        // random initialize some starting coefficients
        std::array<float, 36> coefficients;
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
            std::array<float, 36> gradient;
            CalculateGradient(gradient, coefficients, train, columnIndices, salesIndex);

            // descend
            for (size_t index = 0; index < gradient.size(); ++index)
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
    printf("  Best coefficients are from %zu:%zu\n", bestCoefficientsPopulationIndex, bestCoefficientsStepIndex);
    int i = -1;
    for (float f : bestCoefficients)
    {
        i++;
        printf("    [%i]: %0.4f\n", i, f);
    }
    printf("  test/train R^2 = %f  %f\n", RSquared(bestCoefficients, test, columnIndices, salesIndex), RSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  test/train Adjusted R^2 = %f  %f\n", AdjustedRSquared(bestCoefficients, test, columnIndices, salesIndex), AdjustedRSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
