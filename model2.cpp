#include "utils.h"

/*

Model 1:

f(x) = C

x is Outlet_Location_Type
C is a constant.  C is the mean sales for that location type

*/

void Model2(const CSV& train, const CSV& test)
{
    // get the columns of interest
    printf(__FUNCTION__ "() - use mean sales as a prediction\n");
    int salesIndex = train.GetHeaderIndex("Item_Outlet_Sales");
    if (salesIndex == -1 || test.GetHeaderIndex("Item_Outlet_Sales") != salesIndex)
    {
        printf("Couldn't find Item_Outlet_Sales column.\n");
        return;
    }

    int locationType1Index = train.GetHeaderIndex("Outlet_Location_Type_Tier 1");
    if (locationType1Index == -1 || test.GetHeaderIndex("Outlet_Location_Type_Tier 1") != locationType1Index)
    {
        printf("Couldn't find Outlet_Location_Type_Tier 1 column.\n");
        return;
    }

    int locationType2Index = train.GetHeaderIndex("Outlet_Location_Type_Tier 2");
    if (locationType2Index == -1 || test.GetHeaderIndex("Outlet_Location_Type_Tier 2") != locationType2Index)
    {
        printf("Couldn't find Outlet_Location_Type_Tier 2 column.\n");
        return;
    }

    int locationType3Index = train.GetHeaderIndex("Outlet_Location_Type_Tier 3");
    if (locationType3Index == -1 || test.GetHeaderIndex("Outlet_Location_Type_Tier 3") != locationType3Index)
    {
        printf("Couldn't find Outlet_Location_Type_Tier 3 column.\n");
        return;
    }

    // calculate average sales from training data for each Outlet_Location_Type
    std::unordered_map<float, Average> averageSalesMap;
    for (const auto& row : train.data)
    {
        float locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        averageSalesMap[locationType].AddSample(row[salesIndex]);
    }

    // calculate mean squared error (average squared error) and root mean squared error from training data
    Average Train_MSE;
    std::unordered_map<float, Average> Train_MSEs;
    for (const auto& row : train.data)
    {
        float locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        float error = row[salesIndex] - averageSalesMap[locationType].average;
        Train_MSEs[locationType].AddSample(error * error);
        Train_MSE.AddSample(error * error);
    }
    float Train_RMSE = sqrtf(Train_MSE.average);

    // calculate mean squared error (average squared error) and root mean squared error from test data
    Average Test_MSE;
    std::unordered_map<float, Average> Test_MSEs;
    for (const auto& row : test.data)
    {
        float locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        float error = row[salesIndex] - averageSalesMap[locationType].average;
        Test_MSEs[locationType].AddSample(error * error);
        Test_MSE.AddSample(error * error);
    }
    float Test_RMSE = sqrtf(Test_MSE.average);

    // report results
    for (const auto it : averageSalesMap)
    {
        printf("  Item_Outlet_Sales %i  (%i samples)\n", (int)it.first, it.second.samples);
        printf("    Mean of Item_Outlet_Sales: %0.2f\n", it.second.average);
        printf("    RMSE on training set: %0.2f\n", sqrtf(Train_MSEs[it.first].average));
        printf("    RMSE on test set: %0.2f\n", sqrtf(Test_MSEs[it.first].average));
    }
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
