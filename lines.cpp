#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

int WIDTH;
int HEIGHT;
int gridSize;
bool normalize;


struct Line
{
    Line(double x1, double y1, double x2, double y2):
        x1(x1), y1(y1), x2(x2), y2(y2)
    {}

    double x1;
    double y1;
    double x2;
    double y2;
};

typedef cv::Mat_<uint8_t> Image;


void calculateDescriptor(Image& mat1, Image& mat2, std::vector<int>& results)
{
    int rows = mat1.rows;
    int cols = mat1.cols;

    for (int yy = 0; yy < HEIGHT - gridSize; yy += gridSize)
    {
        for (int xx = 0; xx < WIDTH - gridSize; xx += gridSize)
        {
            int numHorizontalStart = 0;
            int numHorizontalEnd = 0;
            int numVerticalStart = 0;
            int numVerticalEnd = 0;

            for (int y = 0; y < gridSize; ++y)
            {
                int yyy = y + yy;
                if (yyy == 0) continue;
                else if (yyy >= rows) break;

                uint8_t *rowPtr1 = mat1[yyy];
                uint8_t *prevRowPtr1 = mat1[yyy-1];
                uint8_t *rowPtr2 = mat2[yyy];
                for (int x = 0; x < gridSize; ++x)
                {
                    int xxx = x + xx;
                    if (xxx >= cols) break;

                    if (rowPtr1[xxx] && !prevRowPtr1[xxx])
                        numHorizontalStart++;
                    else if (!rowPtr1[xxx] && prevRowPtr1[xxx])
                        numHorizontalEnd++;

                    else if (xxx == 0) continue;
                    if (rowPtr2[xxx] && !rowPtr2[xxx-1])
                        numVerticalStart++;
                    else if (!rowPtr2[xxx] && rowPtr2[xxx-1])
                        numVerticalEnd++;
                }
            }

            results.push_back(numHorizontalStart);
            results.push_back(numHorizontalEnd);
            results.push_back(numVerticalStart);
            results.push_back(numVerticalEnd);
        }
    }
}


void getFeatures(const std::string& filename, std::vector<int>& features, const std::string& imgFilename)
{
    Image img = cv::imread(filename.c_str(), 0);
    if (!img.data)
        throw std::invalid_argument("file not found: " + filename);

    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    int rows = img.rows;
    int cols = img.cols;
    int size = rows * cols;
    int minLineLength = std::min(rows, cols) * 0.035;

    int minX = 99999;
    int minY = 99999;
    int maxX = 0;
    int maxY = 0;

    // find all the horizontal lines
    std::vector<Line> horizontalLines;
    for (int y = 100; y < rows-100; ++y)
    {
        const uint8_t *prevRowPtr = img[y-1];
        const uint8_t *rowPtr = img[y];
        const uint8_t *nextRowPtr = img[y+1];

        int start = 0;
        bool isLine = false;
        for (int x = 100; x < cols-100; ++x)
        {
            bool isDark = rowPtr[x] || prevRowPtr[x] || nextRowPtr[x];
            if (!isLine && isDark)
            {
                start = x;
                isLine =  true;
            }
            else if (isLine && !isDark)
            {
                isLine = false;
                if (x - start > minLineLength)
                {
                    horizontalLines.push_back(Line(start, y, x, y));
                    if (start < minX) minX = start;
                    if (x > maxX) maxX = x;
                }
            }
        }
        if (isLine && cols - 100 - start > minLineLength)
        {
            horizontalLines.push_back(Line(start, y, cols-100, y));
            if (start < minX) minX = start;
            maxX = cols-100;
        }
    }

    // find all the vertical lines
    std::vector<Line> verticalLines;
    std::vector<uint8_t> isLines(cols);
    std::vector<int> starts(cols);
    for (int y = 100; y < rows-100; ++y)
    {
        const uint8_t *rowPtr = img[y];

        for (int x = 100; x < cols-100; ++x)
        {
            bool isDark = rowPtr[x] || rowPtr[x-1] || rowPtr[x+1];
            uint8_t& isLine = isLines[x];
            int& start = starts[x];
            if (!isLine && isDark)
            {
                start = y;
                isLine = 1;
            }
            else if (isLine && !isDark)
            {
                isLines[x] = 0;
                if (y - start > minLineLength)
                {
                    verticalLines.push_back(Line(x, start, x, y));
                    if (start < minY) minY = start;
                    if (y > maxY) maxY = y;
                }
            }
        }
    }
    for (int x = 100; x < cols-100; ++x)
    {
        if (isLines[x] && rows - 100 - starts[x] > minLineLength)
        {
            verticalLines.push_back(Line(x, starts[x], x, rows-100));
            if (starts[x] < minY) minY = starts[x];
            maxY = rows-100;
        }
    }

    double horizontalRatio = (double)WIDTH / (maxX - minX);
    double verticalRatio = (double)HEIGHT / (maxY - minY);

    if (normalize)
    {
        for (int i = 0; i < horizontalLines.size(); ++i)
        {
            horizontalLines[i].x1 -= minX;
            horizontalLines[i].x2 -= minX;
            horizontalLines[i].y1 -= minY;
            horizontalLines[i].y2 -= minY;
            horizontalLines[i].x1 *= horizontalRatio;
            horizontalLines[i].x2 *= horizontalRatio;
            horizontalLines[i].y1 *= verticalRatio;
            horizontalLines[i].y2 *= verticalRatio;
        }
        for (int i = 0; i < verticalLines.size(); ++i)
        {
            verticalLines[i].x1 -= minX;
            verticalLines[i].x2 -= minX;
            verticalLines[i].y1 -= minY;
            verticalLines[i].y2 -= minY;
            verticalLines[i].x1 *= horizontalRatio;
            verticalLines[i].x2 *= horizontalRatio;
            verticalLines[i].y1 *= verticalRatio;
            verticalLines[i].y2 *= verticalRatio;
        }
    }
    else
    {
        for (int i = 0; i < horizontalLines.size(); ++i)
        {
            horizontalLines[i].x1 -= minX;
            horizontalLines[i].x2 -= minX;
            horizontalLines[i].y1 -= minY;
            horizontalLines[i].y2 -= minY;
        }
        for (int i = 0; i < verticalLines.size(); ++i)
        {
            verticalLines[i].x1 -= minX;
            verticalLines[i].x2 -= minX;
            verticalLines[i].y1 -= minY;
            verticalLines[i].y2 -= minY;
        }
    }

    std::vector<Line> actualLines;

    // draw the lines to empty canvases
    Image canvas1 = Image::zeros(rows, cols);
    Image canvas2 = Image::zeros(rows, cols);
    cv::Scalar color(255, 255, 255);
    for (int i = 0; i < horizontalLines.size(); ++i)
    {
        const Line& line = horizontalLines[i];
        if (line.y2 <= maxY)
            cv::line(canvas1,
                     cv::Point((int)line.x1, (int)line.y1),
                     cv::Point((int)line.x2, (int)line.y2),
                     color);
    }

    for (int i = 0; i < verticalLines.size(); ++i)
    {
        const Line& line = verticalLines[i];
        if (line.x2 <= maxX)
            cv::line(canvas2,
                     cv::Point((int)line.x1, (int)line.y1),
                     cv::Point((int)line.x2, (int)line.y2),
                     color);
    }

//    cv::imwrite("/Users/huipeng/Desktop/bw/" + imgFilename, canvas1);

    calculateDescriptor(canvas1, canvas2, features);


//    getchar();

//    cv::namedWindow("foo");
//    cv::imshow("foo", img);
//    cv::waitKey();
}


int main(int argc, char* argv[])
{
    using namespace std;
    
    // read parameters
    if (argc != 5)
        return 0;
    HEIGHT = atoi(argv[1]);
    WIDTH = atoi(argv[2]);
    gridSize = atoi(argv[3]);
    normalize = atoi(argv[4]);

    // calculate and print features
    string imgFilename;
    while (getline(cin, imgFilename))
    {
        vector<int> features;
        getFeatures(imgFilename, features, imgFilename);
        for (int i = 0; i < features.size()/2; ++i)
            cout << features[i] << ' ';
        cout << '\n';
    }

    return 0;
}
