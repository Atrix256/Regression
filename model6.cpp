// This is for central differences, for calculating the gradient
static const float c_epsilon = 0.01f;

// The learning rate, for gradient descent
static const float c_learningRate = 0.01f;

// how many steps of gradient descent are done
static const size_t c_gradientDescentSteps = 1000;

// how many times should it pick a random set of parameters and do gradient descent?
static const size_t c_population = 100;

#include "utils.h"
#include "quadraticfit.h"
#include <array>
#include <random>

/*

Model 3:

f(x,y) = Ax^2 + Bx + Cy^2 + Dy + E

x and y are establishment year and MRP (price)
A, B, C, D and E are coefficients that were learned through gradient descent.

*/

void Model6(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - Quadratic fit of Item_Outlet_Sales based on Outlet_Establishment_Year and Item_MRP\n");

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
    // NOTE: this does the same random numbers every program run, so is deterministic, as written.
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::array<int, 2> columnIndices = { yearIndex, MRPIndex };

    float bestLoss = FLT_MAX;
    std::array<float, 5> bestCoefficients;
    size_t bestCoefficientsPopulationIndex = 0;
    size_t bestCoefficientsStepIndex = 0;

    // for each member of the population
    for (size_t populationIndex = 0; populationIndex < c_population; ++populationIndex)
    {
        // random initialize some starting coefficients
        std::array<float, 5> coefficients;
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
            std::array<float, 5> gradient;
            CalculateGradient(gradient, coefficients, train, columnIndices, salesIndex);

            // descend
            for (size_t index = 0; index < gradient.size(); ++index)
                coefficients[index] -= gradient[index] * c_learningRate;

            // keep the best coefficients seen
            loss = LossFunction(coefficients, train, columnIndices, salesIndex);
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
/*
TODO: best answer is at 46:0. that means gradient descent isn't improving things?!
TODO: model6 is quadratic, model7 is cubic.
TODO: 8,9,10 = ridge, lasso, elastic of quadratic?
*/