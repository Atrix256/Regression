#include <stdio.h>

#include <vector>
#include <string>

struct CSV
{
    std::vector<std::string> headers;
    std::vector<std::vector<float>> data;
};

bool GetNextToken(const char*& cursor, std::string& token, bool& EOL)
{
    // get the string
    const char* end = cursor;
    token = "";
    while (*end != 0 && *end != ',' && *end != '\n' && *end != '\r')
    {
        token += *end;
        end++;
    }

    // report back if this is an end of line or not
    EOL = (*end == '\n' || *end == '\r');

    // skip the separation values
    if (*end == ',')
    {
        end++;
    }
    else
    {
        while (*end == '\n' || *end == '\r')
            end++;
    }

    // move the cursor
    cursor = end;

    // return true if this isn't the end of the data
    return *end != 0;
}

bool LoadCSV(const char* fileName, CSV& csv)
{
    // open the file
    FILE* file = nullptr;
    fopen_s(&file, fileName, "rb");
    if (!file)
        return false;

    // read the file data in and close the file
    fseek(file, 0, SEEK_END);
    std::vector<char> fileData(ftell(file));
    fseek(file, 0, SEEK_SET);
    fread(fileData.data(), fileData.size(), 1, file);
    fclose(file);

    // parse the file
    const char* start = fileData.data();
    std::string nextToken;
    bool EOL = false;
    bool didHeaders = false;
    bool lastTokenWasEOL = false;
    while (GetNextToken(start, nextToken, EOL))
    {
        if (lastTokenWasEOL)
            csv.data.resize(csv.data.size() + 1);

        if (!didHeaders)
        {
            csv.headers.push_back(nextToken);
            didHeaders = EOL;
        }
        else
        {
            float value = 0.0f;
            if (sscanf_s(nextToken.c_str(), "%f", &value) == 1)
                csv.data.rbegin()->push_back(value);
        }

        lastTokenWasEOL = EOL;
    }

    // make sure we have rectangular shaped data
    int rowIndex = 0;
    for (const auto& row : csv.data)
    {
        rowIndex++;
        if (row.size() != csv.headers.size())
            return false;
    }

    // return success
    return true;
}

int main(int argc, char** argv)
{
    CSV testData;
    if (!LoadCSV("data/test.csv", testData))
    {
        printf("could not load data/test.csv");
        return 1;
    }

    CSV trainData;
    if (!LoadCSV("data/train.csv", trainData))
    {
        printf("could not load data/train.csv");
        return 1;
    }

    return 0;
}

/*

BLOG:
* make a nate dawg and warren g meme but "Regularizators... mount up"
* there don't seem to be any nulls in the data, unlike what the article said. not sure why.

*/