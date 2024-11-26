#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

// Funzione per stampare le maschere
void printMasks(Mat& src, uchar labels){

    String name;
    Mat tmp = Mat::zeros(src.rows, src.cols, CV_8U);
    
    for (uchar l = 1; l < labels; l++){
        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                if (src.at<uchar>(i, j) == l){
                    tmp.at<uchar>(i, j) = 255;
                }
            }
        }
        name += to_string((int)l);
        imshow(name, tmp);
        waitKey(0);
        destroyAllWindows();
        name.pop_back();
        tmp -= tmp;
    }

    // Regione "don't care"
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            if (src.at<uchar>(i, j) == 255){
                tmp.at<uchar>(i, j) = 255;
            }
        }
    }
    imshow("don't care", tmp);
    waitKey(0);
}

// Funzione per calcolare l'area della maschera
int getMaskArea(Mat& mask){
    int area = 0;
    for (int i = 0; i < mask.rows; i++)
        for (int j = 0; j < mask.cols; j++)
            if (mask.at<uchar>(i, j) != 0)
                area++;
    return area;
}

// Funzione per espandere la regione
void grow(Mat& src, Mat& dst, Mat& mask, Point seed, int th){

    const Point pointShift2D[8] = {
        Point(-1, -1), Point(-1, 0), Point(-1, 1),
        Point(0, -1), Point(0, 1),
        Point(1, -1), Point(1, 0), Point(1, 1)
    };

    stack<Point> point_stack;
    point_stack.push(seed);

    Point ePoint; // Punto stimato
    Point center;
    
    while (!point_stack.empty()){
        center = point_stack.top();
        mask.at<uchar>(center) = 1;
        point_stack.pop();

        for (int i = 0; i < 8; i++){
            ePoint = center + pointShift2D[i];
            
            if (ePoint.x < 0 || ePoint.x > src.cols - 1 ||
                ePoint.y < 0 || ePoint.y > src.rows - 1){
                continue;
            } else{
                int delta = int(pow(src.at<uchar>(center) - src.at<uchar>(ePoint), 2));
                
                if (dst.at<uchar>(ePoint) == 0 && mask.at<uchar>(ePoint) == 0 && delta < th){
                    mask.at<uchar>(ePoint) = 1;
                    point_stack.push(ePoint);
                }
            }
        }
    }
}

// Funzione principale di crescita regionale
uchar regionGrowing(Mat& src, Mat& dst, double minSizeRegFactor, int maxRegNum, int th){

    int minSizeReg = minSizeRegFactor * src.rows * src.cols;
    uchar label = 1;
    dst = Mat::zeros(src.rows, src.cols, CV_8U);
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8U);
    int maskArea;

    for (int x = 0; x < src.cols; x++){
        for (int y = 0; y < src.rows; y++)
            if (dst.at<uchar>(Point(x, y)) == 0){
                grow(src, dst, mask, Point(x, y), th);
                maskArea = getMaskArea(mask);
                
                if (maskArea > minSizeReg){
                    dst += mask * label;
                    label++;
                    
                    if (label > maxRegNum)
                        exit(-2); // Sovrapposizione
                } else {
                    dst += mask * 255;
                }
                mask -= mask;
            }

        int i = 0; // Barra di caricamento
        if (i++ > src.cols / 10){
            cout << "Fatto: " << (x * 100) / src.cols << "%" << endl;
            i = 0;
        }
    }
    
    cout << (int)label << endl;
    return label;
}

// Funzione principale
int main(int argc, char** argv){

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    double minSizeRegFactor = 0.01f;
    int maxRegNum = 50;
    int th = 15;
    uchar labels = regionGrowing(src, dst, minSizeRegFactor, maxRegNum, th);
    printMasks(dst, labels);

    waitKey(0);
    return 0;
}
