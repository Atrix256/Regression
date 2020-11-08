#include "utils.h"

/*

Model 1:

f() = C

C is a constant.  C is the mean sales.

*/

void Model1(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - use mean sales as a prediction\n");

    // find out which column is the sales
    int salesIndex = train.GetHeaderIndex("Item_Outlet_Sales");
    if (salesIndex == -1 || test.GetHeaderIndex("Item_Outlet_Sales") != salesIndex)
    {
        printf("Couldn't find Item_Outlet_Sales column.\n");
        return;
    }

    // calculate average sales from training data
    Average averageSales;
    for (const auto& row : train.data)
        averageSales.AddSample(row[salesIndex]);

    // calculate mean squared error (average squared error) and root mean squared error from training data
    Average Train_MSE;
    for (const auto& row : train.data)
    {
        float error = row[salesIndex] - averageSales.average;
        Train_MSE.AddSample(error * error);
    }
    float Train_RMSE = sqrtf(Train_MSE.average);

    // calculate mean squared error (average squared error) and root mean squared error from test data
    Average Test_MSE;
    for (const auto& row : test.data)
    {
        float error = row[salesIndex] - averageSales.average;
        Test_MSE.AddSample(error * error);
    }
    float Test_RMSE = sqrtf(Test_MSE.average);

    // report results
    printf("  Mean of Item_Outlet_Sales: %0.2f\n", averageSales.average);
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
