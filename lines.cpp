#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <boost/chrono/chrono_io.hpp>

#define WIDTH 2880
#define HEIGHT 3520


struct Line
{
    Line(int x1, int y1, int x2, int y2):
        x1(x1), y1(y1), x2(x2), y2(y2)
    {}

    int x1;
    int y1;
    int x2;
    int y2;
};

typedef cv::Mat_<uint8_t> Image;


void calculateDescriptor(Image& mat1, Image& mat2, std::vector<int>& results)
{
    int rows = mat1.rows;
    int cols = mat1.cols;
    const int gridSize = 64;

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


Image dilateVertical(Image& mat, uint8_t val = 255)
{
    int rows = mat.rows;
    int cols = mat.cols;
    Image result = Image::zeros(rows, cols);


    for (int x = 0; x < cols; ++x)
    {
        if (mat(0, x))
        {
            result(0, x) = val;
            result(1, x) = val;
        }
        if (mat(rows-1, x))
        {
            result(rows-1, x) = val;
            result(rows-2, x) = val;
        }
    }

    for (int y = 1; y < rows - 1; ++y)
    {
        uint8_t *srcRowPtr = mat[y];
        uint8_t *destRowPtr = result[y];
        uint8_t *destPrevRowPtr = result[y-1];
        uint8_t *destNextRowPtr = result[y+1];
        for (int x = 0; x < cols; ++x)
        {
            if (srcRowPtr[x])
            {
                destPrevRowPtr[x] = val;
                destRowPtr[x] = val;
                destNextRowPtr[x] = val;
            }
        }
    }

    return result;
}

Image dilateHorizontal(Image& mat, uint8_t val = 255)
{
    int rows = mat.rows;
    int cols = mat.cols;
    Image result = Image::zeros(rows, cols);

    for (int y = 0; y < rows; ++y)
    {
        uint8_t *srcRowPtr = mat[y];
        uint8_t *destRowPtr = result[y];
        if (srcRowPtr[0])
        {
            destRowPtr[0] = val;
            destRowPtr[1] = val;
        }
        for (int x = 1; x < cols - 1; ++x)
        {
            if (srcRowPtr[x])
            {
                destRowPtr[x-1] = val;
                destRowPtr[x] = val;
                destRowPtr[x+1] = val;
            }
        }
        if (srcRowPtr[cols-1])
        {
            destRowPtr[cols-1] = val;
            destRowPtr[cols-2] = val;
        }
    }

    return result;
}


void getFeatures(const std::string& filename, std::vector<int>& features)
{
    Image img = cv::imread(filename.c_str(), 0);
    if (!img.data)
        throw std::invalid_argument("file not found: " + filename);

    int rows = img.rows;
    int cols = img.cols;
    int size = rows * cols;
    int minLineLength = std::min(rows, cols) * 0.035;

    boost::chrono::high_resolution_clock::time_point t0;
    boost::chrono::high_resolution_clock::time_point t1;

    // invert image
    t0 = boost::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i)
        img.data[i] ^= 255;
    std::cout << "1:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';

    // find all the horizontal lines
    t0 = boost::chrono::high_resolution_clock::now();
    std::vector<Line> horizontalLines;
    Image dilatedImg = dilateVertical(img);
    for (int y = 100; y < rows-100; ++y)
    {
        const uint8_t *rowPtr = dilatedImg[y];

        int start = 0;
        bool isLine = false;
        for (int x = 100; x < cols-100; ++x)
        {
            uint8_t el = rowPtr[x];
            if (!isLine && el)
            {
                start = x;
                isLine =  true;
            }
            else if (isLine && !el)
            {
                isLine = false;
                if (x - start > minLineLength)
                    horizontalLines.push_back(Line(start, y, x, y));
            }
        }
        if (isLine && cols - 100 - start > minLineLength)
            horizontalLines.push_back(Line(start, y, cols-100, y));
    }
    std::cout << "2:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';

    // find all the vertical lines
    t0 = boost::chrono::high_resolution_clock::now();
    std::vector<Line> verticalLines;
    dilatedImg = dilateHorizontal(img);
    std::cout << "2.5:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';
    std::vector<uint8_t> isLines(cols);
    std::vector<int> starts(cols);
    for (int y = 100; y < rows-100; ++y)
    {
        const uint8_t *rowPtr = dilatedImg[y];

        for (int x = 100; x < cols-100; ++x)
        {
            uint8_t el = rowPtr[x];
            if (!isLines[x] && el)
            {
                starts[x] = y;
                isLines[x] = 1;
            }
            else if (isLines[x] && !el)
            {
                isLines[x] = 0;
                if (y - starts[x] > minLineLength)
                    verticalLines.push_back(Line(x, starts[x], x, y));
            }
        }
    }
    for (int x = 100; x < cols-100; ++x)
    {
        if (isLines[x] && rows - 100 - starts[x] > minLineLength)
            verticalLines.push_back(Line(x, starts[x], x, rows-100));
    }
    std::cout << "3:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';

    t0 = boost::chrono::high_resolution_clock::now();
    int minX = 99999;
    int minY = 99999;
    int maxX = 0;
    int maxY = 0;
    for (int i = 0; i < horizontalLines.size(); ++i)
    {
        minX = std::min(minX, horizontalLines[i].x1);
        maxX = std::max(maxX, horizontalLines[i].x2);
    }
    for (int i = 0; i < verticalLines.size(); ++i)
    {
        minY = std::min(minY, verticalLines[i].y1);
        maxY = std::max(maxY, verticalLines[i].y2);
    }

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
    std::cout << "4:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';

    std::vector<Line> actualLines;

    // draw the lines to empty canvases
    t0 = boost::chrono::high_resolution_clock::now();
    Image canvas1 = Image::zeros(rows, cols);
    Image canvas2 = Image::zeros(rows, cols);
    cv::Scalar color(255, 255, 255);
    for (int i = 0; i < horizontalLines.size(); ++i)
    {
        const Line& line = horizontalLines[i];
        if (line.y2 <= maxY)
            cv::line(canvas1,
                     cv::Point(line.x1, line.y1),
                     cv::Point(line.x2, line.y2),
                     color);
    }

    for (int i = 0; i < verticalLines.size(); ++i)
    {
        const Line& line = verticalLines[i];
        if (line.x2 <= maxX)
            cv::line(canvas1,
                     cv::Point(line.x1, line.y1),
                     cv::Point(line.x2, line.y2),
                     color);
    }
    std::cout << "5:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';

    t0 = boost::chrono::high_resolution_clock::now();
    calculateDescriptor(canvas1, canvas2, features);
    std::cout << "6:\t" << boost::chrono::high_resolution_clock::now() - t0 << '\n';


    getchar();

//    cv::namedWindow("foo");
//    cv::imshow("foo", canvas2);
//    cv::waitKey();
}


int main(int argc, char* argv[])
{
    using namespace std;

    string srcFolder = "/Users/huipeng/EO990RW8/";

    ifstream firstPagesFile("/Users/huipeng/EO990RW8/first_pages.txt", ifstream::in);
    ofstream linesFile("/Users/huipeng/EO990RW8/lines_extract.txt", ostream::out);
    string imgFilename;
    while (getline(firstPagesFile, imgFilename))
    {
        try
        {
            vector<int> features;
            getFeatures(srcFolder + imgFilename, features);
            linesFile << imgFilename << " \n";
            for (int i = 0; i < features.size(); ++i)
                linesFile << features[i] << ' ';
            linesFile << '\n';
        }
        catch (exception& e)
        { }
    }
    

    return 0;
}
