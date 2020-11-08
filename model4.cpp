#include "utils.h"
#include "linearfit.h"
#include <random>

/*

Model 4:

f(x,y) = Ax + By + Cz + D

x and y are establishment year, MRP (price) and item weight
A, B, C and D are coefficients that were learned through gradient descent.

*/


void Model4(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - Linear fit of Item_Outlet_Sales based on Outlet_Establishment_Year, Item_MRP and Item_Weight\n");

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

    int WeightIndex = train.GetHeaderIndex("Item_Weight");
    if (WeightIndex == -1 || test.GetHeaderIndex("Item_Weight") != WeightIndex)
    {
        printf("Couldn't find Item_Weight column.\n");
        return;
    }

    // do gradient descent
    // NOTE: this does the same random numbers every program run, so is deterministic, as written.
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::array<int, 3> columnIndices = { yearIndex, MRPIndex, WeightIndex };

    float bestLoss = FLT_MAX;
    std::array<float, 4> bestCoefficients;
    size_t bestCoefficientsPopulationIndex = 0;
    size_t bestCoefficientsStepIndex = 0;

    // for each member of the population
    for (size_t populationIndex = 0; populationIndex < c_population; ++populationIndex)
    {
        // random initialize some starting coefficients
        std::array<float, 4> coefficients;
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
            std::array<float, 4> gradient;
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
    printf("  Best coefficients = %0.4f, %0.4f, %0.4f, %0.4f  (from %zu:%zu)\n", bestCoefficients[0], bestCoefficients[1], bestCoefficients[2], bestCoefficients[3], bestCoefficientsPopulationIndex, bestCoefficientsStepIndex);
    printf("  test/train R^2 = %f  %f\n", RSquared(bestCoefficients, test, columnIndices, salesIndex), RSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  test/train Adjusted R^2 = %f  %f\n", AdjustedRSquared(bestCoefficients, test, columnIndices, salesIndex), AdjustedRSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}

