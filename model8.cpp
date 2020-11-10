// This is for central differences, for calculating the gradient
static const double c_epsilon = 0.001f;

// The learning rate, for gradient descent
static double c_learningRate = 1.0f;

// how many steps of gradient descent are done
static const size_t c_gradientDescentSteps = 1000;

// how many times should it pick a random set of parameters and do gradient descent?
static const size_t c_population = 100;

// L2 regularization alpha value
static const double c_L1RegAlpha = 0.0f;
static const double c_L2RegAlpha = 1000.0f;

#include "utils.h"
#include "quadraticfit.h"
#include <array>
#include <random>

/*

Model 8:  Model 6 with ridge (L2) regression

f(x,y) = Ax^2 + Bx + Cy^2 + Dy + E

x and y are establishment year and MRP (price)
A, B, C, D and E are coefficients that were learned through gradient descent.

Ridge regression (L2 regression) means the square of the coefficients times an alpha is added into the MSE to promote smaller coefficients

*/

void Model8(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - Quadratic fit of Item_Outlet_Sales based on Outlet_Establishment_Year and Item_MRP, with Ridge (L2) Reg\n");

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
    std::uniform_real_distribution<double> dist(-10.0f, 10.0f);
    std::array<int, 2> columnIndices = { yearIndex, MRPIndex };

    double bestLoss = FLT_MAX;
    std::array<double, 5> bestCoefficients;
    size_t bestCoefficientsPopulationIndex = 0;
    size_t bestCoefficientsStepIndex = 0;

    // for each member of the population
    for (size_t populationIndex = 0; populationIndex < c_population; ++populationIndex)
    {
        // random initialize some starting coefficients
        std::array<double, 5> coefficients;
        for (double& f : coefficients)
            f = dist(rng);

        // keep the best coefficients seen
        double loss = LossFunction(coefficients, train, columnIndices, salesIndex, c_L1RegAlpha, c_L2RegAlpha);
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
            std::array<double, 5> gradient;
            CalculateGradient(gradient, coefficients, train, columnIndices, salesIndex, c_L1RegAlpha, c_L2RegAlpha);

            // do gradient descent with an adaptive learning rate to make sure it isn't increasing the loss function
            bool newLossWasLarger = false;
            double newLoss = 0.0f;
            std::array<double, 5> newCoefficients;
            do
            {
                // descend
                for (size_t index = 0; index < gradient.size(); ++index)
                    newCoefficients[index] = coefficients[index] - gradient[index] * c_learningRate;

                // keep the best coefficients seen
                newLoss = LossFunction(newCoefficients, train, columnIndices, salesIndex, c_L1RegAlpha, c_L2RegAlpha);
                if (newLoss >= loss)
                {
                    c_learningRate /= 10.0f;
                    newLossWasLarger = true;
                }
            }
            while (newLoss >= loss);
            loss = newLoss;
            coefficients = newCoefficients;

            // grow the learning rate for next iteration
            c_learningRate *= 10.0f;

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
    double Train_MSE = LossFunction(bestCoefficients, train, columnIndices, salesIndex, c_L1RegAlpha, c_L2RegAlpha);
    double Test_MSE = LossFunction(bestCoefficients, test, columnIndices, salesIndex, c_L1RegAlpha, c_L2RegAlpha);

    // Report results
    printf("  Best coefficients are from %zu:%zu\n", bestCoefficientsPopulationIndex, bestCoefficientsStepIndex);
    int i = -1;
    for (double f : bestCoefficients)
    {
        i++;
        printf("    [%i]: %0.4f\n", i, f);
    }
    printf("  test/train R^2 = %f  %f\n", RSquared(bestCoefficients, test, columnIndices, salesIndex), RSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  test/train Adjusted R^2 = %f  %f\n", AdjustedRSquared(bestCoefficients, test, columnIndices, salesIndex), AdjustedRSquared(bestCoefficients, train, columnIndices, salesIndex));
    printf("  RMSE on training set: %0.2f\n", sqrt(Train_MSE));
    printf("  RMSE on test set: %0.2f\n\n", sqrt(Test_MSE));
}
