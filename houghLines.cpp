//https://github.com/eToTheEcs/hough-transform/blob/master/hough.cpp

#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

// Funzione per convertire coordinate polari in coordinate cartesiane
void polarToCartesian(double rho, int theta, Point& p1, Point& p2){
    int alpha = 1000;

    // Calcola le coordinate cartesiane dei due punti estremi della linea
    int x0 = cvRound(rho*cos(theta));
	int y0 = cvRound(rho*sin(theta));

	p1.x = cvRound(x0 + alpha*(-sin(theta)));
	p1.y = cvRound(y0 + alpha*(cos(theta)));

	p2.x = cvRound(x0 - alpha*(-sin(theta)));
	p2.y = cvRound(y0 - alpha*(cos(theta)));
}

// Funzione per la trasformata di Hough
void houghLines(Mat& src, Mat& dst, int cannyLTH, int cannyHTH, int HoughTH){

    //1 Inizializza una matrice di voti per le linee
    int maxDist = hypot(src.rows, src.cols);
    vector<vector<int>> votes(maxDist*2, vector<int>(180, 0));

    //2 Applica una gaussiana e il rilevamento dei bordi utilizzando Canny
    Mat gsrc, edges;
    GaussianBlur(src,gsrc,Size(3,3),0,0);
    Canny(gsrc,edges,cannyLTH,cannyHTH);

    //3 Calcola la trasformata di Hough accumulando i voti
    double rho;
    int theta;
    for(int x=0; x<edges.rows; x++)
        for(int y=0; y<edges.cols; y++)
            if(edges.at<uchar>(x,y) == 255)
                for(theta = 0; theta <= 180; theta++){
                    // Calcola il parametro rho nella trasformata di Hough
                    rho = round(y*cos(theta-90) + x*sin(theta-90)) + maxDist;
                    votes[static_cast<int>(rho)][theta]++;
                }

    //4 - Disegna le linee trovate sull'immagine di destinazione
    dst=src.clone();
    Point p1, p2;
    for(size_t i=0; i<votes.size(); i++)
        for(size_t j=0; j<votes[i].size(); j++)
            if(votes[i][j] >= HoughTH){
                // Ottieni i parametri rho e theta dalla matrice dei voti
                rho = i-maxDist;
                theta = j-90;
                // Converti le coordinate polari in coordinate cartesiane
                polarToCartesian(rho,theta,p1,p2);
                // Disegna la linea sull'immagine di destinazione
                line(dst,p1,p2,Scalar(0,0,255),2,LINE_AA);
            }
}

int main(int argc, char** argv){

    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    Mat dst;
    // Impostazioni per il rilevamento dei bordi e la trasformata di Hough
    int cannyLTH = 100;
    int cannyHTH = 150;
    int HoughTH = 200;
    houghLines(src,dst,cannyLTH,cannyHTH,HoughTH);


    imshow("src",src);
    imshow("dst",dst);
    waitKey(0);
    return 0;
}
