#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <queue>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>


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

void drawLines(cv::Mat& canvas, const std::vector<Line> lines, unsigned char val=255)
{
    cv::Scalar color(val, val, val);
    for (int i = 0; i < lines.size(); ++i)
        cv::line(canvas,
                 cv::Point(lines[i].x1, lines[i].y1),
                 cv::Point(lines[i].x2, lines[i].y2),
                 color);
}



void groupLines(cv::Mat& canvas, std::vector<Line>& results, bool isHorizontal)
{
    int rows = canvas.rows;
    int cols = canvas.cols;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            unsigned char el = canvas.at<unsigned char>(y, x);
            if (!el) continue;

            int minX = cols;
            int maxX = 0;
            int minY = rows;
            int maxY = 0;
            double averageY = 0.0;
            double averageX = 0.0;
            int numPoints = 0;

            std::queue<cv::Point> fringe;
            fringe.push(cv::Point(x, y));
            canvas.at<unsigned char>(y, x) = 0;

            while (fringe.size())
            {
                cv::Point pt = fringe.front();
                fringe.pop();

                minX = std::min(minX, pt.x);
                maxX = std::max(maxX, pt.x);
                minY = std::min(minY, pt.y);
                maxY = std::max(maxY, pt.y);
                averageX += pt.x;
                averageY += pt.y;
                numPoints++;

                for (int dy = -2; dy <= 2; ++dy)
                {
                    for (int dx = -2; dx <= 2; ++dx)
                    {
                        int xx = pt.x + dx;
                        int yy = pt.y + dy;
                        if (xx < 0 || xx >= cols || yy < 0 || yy >= rows)
                            continue;
                        if (canvas.at<unsigned char>(yy, xx) == 0)
                            continue;

                        canvas.at<unsigned char>(yy, xx) = 0;
                        fringe.push(cv::Point(xx, yy));
                    }
                }
            }

            averageY /= numPoints;
            averageX /= numPoints;

            if (isHorizontal)
                results.push_back(Line(minX, averageY, maxX, averageY));
            else
                results.push_back(Line(averageX, minY, averageX, maxY));
        }
    }
}

void calculateDescriptor(cv::Mat& mat1, cv::Mat& mat2, std::vector<int>& results)
{
    int rows = mat1.rows;
    int cols = mat1.cols;
    const int gridSize = 64;

    for (int yy = 0; yy < 3520 - gridSize; yy += gridSize)
    {
        for (int xx = 0; xx < 2880 - gridSize; xx += gridSize)
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

                unsigned char *rowPtr1 = mat1.ptr<unsigned char>(yyy);
                unsigned char *prevRowPtr1 = mat1.ptr<unsigned char>(yyy-1);
                unsigned char *rowPtr2 = mat2.ptr<unsigned char>(yyy);
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


cv::Mat dilateVertical(cv::Mat& mat, unsigned char val = 255)
{
    int rows = mat.rows;
    int cols = mat.cols;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC1);


    for (int x = 0; x < cols; ++x)
    {
        if (mat.at<unsigned char>(0, x))
        {
            result.at<unsigned char>(0, x) = val;
            result.at<unsigned char>(1, x) = val;
        }
        if (mat.at<unsigned char>(rows-1, x))
        {
            result.at<unsigned char>(rows-1, x) = val;
            result.at<unsigned char>(rows-2, x) = val;
        }
    }

    for (int y = 1; y < rows - 1; ++y)
    {
        unsigned char *srcRowPtr = mat.ptr<unsigned char>(y);
        unsigned char *destRowPtr = result.ptr<unsigned char>(y);
        unsigned char *destPrevRowPtr = result.ptr<unsigned char>(y-1);
        unsigned char *destNextRowPtr = result.ptr<unsigned char>(y+1);
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

cv::Mat dilateHorizontal(cv::Mat& mat, unsigned char val = 255)
{
    int rows = mat.rows;
    int cols = mat.cols;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        unsigned char *srcRowPtr = mat.ptr<unsigned char>(y);
        unsigned char *destRowPtr = result.ptr<unsigned char>(y);
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


std::vector<int> getLines(const std::string& filename)
{
    cv::Mat img = cv::imread(filename.c_str(), 0);
    if (!img.data)
        throw std::invalid_argument("file not found: " + filename);

    int rows = img.rows;
    int cols = img.cols;
    int minLineLength = std::min(rows, cols) * 0.035;

    // invert image
    for (int y = 0; y < rows; ++y)
    {
        unsigned char *rowPtr = img.ptr<unsigned char>(y);
        for (int x = 0; x < cols; ++x)
            rowPtr[x] ^= 255;
    }

    // find all the horizontal lines
    std::vector<Line> horizontalLines;
    cv::Mat dilatedImg = dilateVertical(img);
    for (int y = 100; y < rows-100; ++y)
    {
        const unsigned char *rowPtr = dilatedImg.ptr<unsigned char>(y);

        int start = 0;
        bool isLine = false;
        for (int x = 100; x < cols-100; ++x)
        {
            unsigned char el = rowPtr[x];
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

    // find all the vertical lines
    std::vector<Line> verticalLines;
    dilatedImg = dilateHorizontal(img);
    for (int x = 100; x < cols-100; ++x)
    {
        int start = 0;
        bool isLine = false;
        for (int y = 100; y < rows-100; ++y)
        {
            unsigned char el = dilatedImg.at<unsigned char>(y, x);
            if (!isLine && el)
            {
                start = y;
                isLine =  true;
            }
            else if (isLine && !el)
            {
                isLine = false;
                if (y - start > minLineLength)
                    verticalLines.push_back(Line(x, start, x, y));
            }
        }
        if (isLine && rows - 100 - start > minLineLength)
            verticalLines.push_back(Line(x, start, x, rows-100));
    }

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

    std::vector<Line> actualLines;
    std::vector<int> descriptor;

    // draw the lines to empty canvases
    cv::Mat canvas1 = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat canvas2 = cv::Mat::zeros(rows, cols, CV_8UC1);
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

//    groupLines(canvas1, actualLines, true);
//    groupLines(canvas2, actualLines, false);

    calculateDescriptor(canvas1, canvas2, descriptor);

    // draw the vertical lines on an empty canvas and then group the pixels

//    double ratio = 0.30;
//    int resizedRows = (int)(rows * ratio);
//    int resizedCols = (int)(cols * ratio);
//    cv::Mat canvas2 = cv::Mat::zeros(resizedRows, resizedCols, CV_8UC1);

//    for (int i = 0; i < actualLines.size(); ++i)
//    {
//        cv::line(canvas2,
//                 cv::Point(actualLines[i].x1 * ratio, actualLines[i].y1 * ratio),
//                 cv::Point(actualLines[i].x2 * ratio, actualLines[i].y2 * ratio),
//                 cv::Scalar(255, 255, 255));
//    }
//    for (int i = 0; i < horizontalLines.size(); ++i)
//    {
//        cv::line(canvas2,
//                 cv::Point(horizontalLines[i].x1, horizontalLines[i].y1),
//                 cv::Point(horizontalLines[i].x2, horizontalLines[i].y2),
//                 cv::Scalar(255, 255, 255));
//    }
//    for (int i = 0; i < verticalLines.size(); ++i)
//    {
//        cv::line(canvas2,
//                 cv::Point(verticalLines[i].x1, verticalLines[i].y1),
//                 cv::Point(verticalLines[i].x2, verticalLines[i].y2),
//                 cv::Scalar(255, 255, 255));
//    }

//    cv::namedWindow("foo");
//    cv::imshow("foo", canvas2);
//    cv::waitKey();

    return descriptor;
//    return actualLines;
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
//            vector<Line> lines = getLines(srcFolder + imgFilename);
//            cout << lines.size() << endl;
//            linesFile << imgFilename << ' ' << lines.size() << '\n';
//            for (int i = 0; i < lines.size(); ++i)
//                linesFile << lines[i].x1 << ' ' << lines[i].y1 << ' ' << lines[i].x2 << ' ' << lines[i].y2 << '\n';

            vector<int> descriptor = getLines(srcFolder + imgFilename);
            cout << descriptor.size() << endl;
            linesFile << imgFilename << " \n";
            for (int i = 0; i < descriptor.size(); ++i)
                linesFile << descriptor[i] << ' ';
            linesFile << '\n';
        }
        catch (exception& e)
        {
        }
    }
    

    return 0;
}
