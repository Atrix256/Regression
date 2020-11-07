#include "utils.h"

/*

Model 1:

f(x) = C

where C is a constant.  C is the mean sales.

*/

void Model1(const CSV& train, const CSV& test)
{
    // find out which column is the sales
    printf(__FUNCTION__ "() - use mean sales as a prediction\n");
    int salesIndex = train.GetHeaderIndex("Item_Outlet_Sales");
    if (salesIndex == -1 || test.GetHeaderIndex("Item_Outlet_Sales") != salesIndex)
    {
        printf("Couldn't find Item_Outlet_Sales column.\n");
        return;
    }

    // calculate average sales from training data
    float averageSales = 0.0f;
    int index = 0;
    for (const auto& row : train.data)
    {
        index++;
        averageSales = Lerp(averageSales, row[salesIndex], 1.0f / float(index));
    }

    // calculate mean squared error (average squared error) and root mean squared error from training data
    float Train_MSE = 0.0f;
    index = 0;
    for (const auto& row : train.data)
    {
        index++;
        float error = row[salesIndex] - averageSales;
        error *= error;
        Train_MSE = Lerp(Train_MSE, error, 1.0f / float(index));
    }
    float Train_RMSE = sqrtf(Train_MSE);


    // calculate mean squared error (average squared error) and root mean squared error from test data
    float Test_MSE = 0.0f;
    index = 0;
    for (const auto& row : test.data)
    {
        index++;
        float error = row[salesIndex] - averageSales;
        error *= error;
        Test_MSE = Lerp(Test_MSE, error, 1.0f / float(index));
    }
    float Test_RMSE = sqrtf(Test_MSE);

    // report results
    printf("  Mean of Item_Outlet_Sales: %0.2f\n", averageSales);
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
