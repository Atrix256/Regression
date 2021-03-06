#include "utils.h"

/*

Model 2:

f(x) = C

x is Outlet_Location_Type
C is a constant.  C is the mean sales for that location type

*/

void Model2(const CSV& train, const CSV& test)
{
    printf(__FUNCTION__ "() - use mean sales per Outlet_Location_Type as a prediction\n");

    // get the columns of interest
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
    std::unordered_map<double, Average> averageSalesMap;
    for (const auto& row : train.data)
    {
        double locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        averageSalesMap[locationType].AddSample(row[salesIndex]);
    }

    // calculate mean squared error (average squared error) and root mean squared error from training data
    Average Train_MSE;
    std::unordered_map<double, Average> Train_MSEs;
    for (const auto& row : train.data)
    {
        double locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        double error = row[salesIndex] - averageSalesMap[locationType].average;
        Train_MSEs[locationType].AddSample(error * error);
        Train_MSE.AddSample(error * error);
    }
    double Train_RMSE = sqrt(Train_MSE.average);

    // calculate mean squared error (average squared error) and root mean squared error from test data
    Average Test_MSE;
    std::unordered_map<double, Average> Test_MSEs;
    for (const auto& row : test.data)
    {
        double locationType = row[locationType1Index] * 4.0f + row[locationType2Index] * 2.0f + row[locationType1Index];
        double error = row[salesIndex] - averageSalesMap[locationType].average;
        Test_MSEs[locationType].AddSample(error * error);
        Test_MSE.AddSample(error * error);
    }
    double Test_RMSE = sqrt(Test_MSE.average);

    // report results
    for (const auto it : averageSalesMap)
    {
        printf("  Item_Outlet_Sales %i  (%i samples)\n", (int)it.first, it.second.samples);
        printf("    Mean of Item_Outlet_Sales: %0.2f\n", it.second.average);
        printf("    RMSE on training set: %0.2f\n", sqrt(Train_MSEs[it.first].average));
        printf("    RMSE on test set: %0.2f\n", sqrt(Test_MSEs[it.first].average));
    }
    printf("  RMSE on training set: %0.2f\n", Train_RMSE);
    printf("  RMSE on test set: %0.2f\n\n", Test_RMSE);
}
