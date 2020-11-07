#include <stdio.h>
#include "utils.h"

int main(int argc, char** argv)
{
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

    Model1(train, test);

    return 0;
}

/*

BLOG:
* make a nate dawg and warren g meme but "Regularizators... mount up"
* there don't seem to be any nulls in the data, unlike what the article said. not sure why.

* Model1 prediction function is f(x) = constant.  The constant is the average sales, not taking anything else into account.
* Model1 - the MSE is different than what the page says, but not different than what the python says if you run the python from the page on the data set!

*/