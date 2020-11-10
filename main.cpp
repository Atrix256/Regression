#include <stdio.h>
#include "utils.h"

int main(int argc, char** argv)
{
    // load the training and test data
    CSV train;
    if (!LoadCSV("data/train.csv", train))
    {
        printf("could not load data/train.csv");
        return 1;
    }

    CSV test;
    if (!LoadCSV("data/test.csv", test))
    {
        printf("could not load data/test.csv");
        return 1;
    }

    // process the Outlet_Establishment_Year column to be smaller numbers, so fewer numerical concerns
    {
        int yearIndex = train.GetHeaderIndex("Outlet_Establishment_Year");
        if (yearIndex == -1 || test.GetHeaderIndex("Outlet_Establishment_Year") != yearIndex)
        {
            printf("Couldn't find Outlet_Establishment_Year column.\n");
            return 1;
        }

        for (auto& row : train.data)
            row[yearIndex] = 2020.0f - row[yearIndex];

        for (auto& row : test.data)
            row[yearIndex] = 2020.0f - row[yearIndex];
    }

    // TODO: TEMP!
#if 0
    Model1(train, test);
    Model2(train, test);
    Model3(train, test);
    Model4(train, test);
    Model5(train, test);
#endif

    Model6(train, test);

    return 0;
}

/*

BLOG:
* make a nate dawg and warren g meme but "Regularizators... mount up"
* there don't seem to be any nulls in the data, unlike what the article said. not sure why.

* Model1 prediction function is f(x) = constant.  The constant is the average sales, not taking anything else into account.
* Model1 - the MSE is different than what the page says, but not different than what the python says if you run the python from the page on the data set!

* Model2 prediction function is same as Model1, but a different function per location type.
* there are 3 columns for location type... tier 1,2,3 which are booleans. made it a bitfield since it's a map key. article didn't really explain what it did.
* having multiple averages instead of a single one is just a more detailed model. like a piecewise constant function!

* Model3 - Linear fit of Item_Outlet_Sales based on Outlet_Establishment_Year and Item_MRP
* MRP is item price
* there are better ways to do gradient descent. Momentum, Adam, do it for multiple populations and pick the best, simulated annealing, mini batches etc.
* if doing multiple and keeping the best, using low discrepancy sequences or blue noise to initialize coefficients would be better than pure random white noise by having better coverage over the sampling space.
* also called "orthogonal initialization"
* Loss function is MSE/2. the 2 doesn't change anything, but whatever :P
*  actually, we'll use RMSE to make smaller magnitude numbers for the same result. less to worry about w/ floating point though.
* get gradient numerically with central differences
* talk about R^2 and adjusted R^2

* Model4 - same, but adding item_weight. didn't seem to help much?

* Model5 - all items. the article talks about one hot encoding. it looks like the data already has that. if you have an enum, it turns it into a bool per item.
* adjusted R squared went down... means the model got more complex w/o benefit.
* lots slower! could probably be optimized.
* fit isn't as good. it seems to want to go deeper (step count) instead of wider (population)

* Model6 - quadratic fit with Outlet_Establishment_Year and Item_MRP
* I had problems with raw year. maybe numbers too large? replaced year column with 2020-year. things cleaned up.

* talk about some good points about variance / bias from that post. like high bias, low variance is underfitting

! you weren't descending with all gradient indices, oops! you fixed it though, so the above info about accuracy may not be correct

*/