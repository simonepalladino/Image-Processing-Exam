#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Funzione per eseguire la rilevazione dei cerchi mediante Hough Circle
void houghCircles(Mat& src, Mat& dst, int cannyLTH, int cannyHTH, int HoughTH, int Rmin, int Rmax){

    //1 Inizializzare un vettore tridimensionale per memorizzare i voti per ciascuna combinazione di pixel e raggio
    vector<vector<vector<int>>>
        votes(src.rows, vector<vector<int>>(src.cols, vector<int>(Rmax-Rmin+1,0)));

    //2 Applicare il filtro di sfocatura Gaussiana e la rilevazione dei bordi Canny
    Mat gsrc, edges;
    GaussianBlur(src,gsrc,Size(7,7),0,0);
    Canny(gsrc,edges,cannyLTH,cannyHTH); //esegue l'algoritmo di rilevazione dei bordi di Canny sull'immagine

    //3 Trasformata di Hough per i cerchi
    for(int x=0; x<edges.rows; x++)
        for(int y=0; y<edges.cols; y++)
            if(edges.at<uchar>(x,y) == 255)
                for(int r=Rmin; r<=Rmax; r++)
                    for(int theta=0; theta<360; theta++){
                        int a = y - r*cos(theta*CV_PI/180);
                        int b = x - r*sin(theta*CV_PI/180);
                        if(a>=0 && a<edges.cols && b>=0 && b<edges.rows)
                            votes[b][a][r-Rmin]++;
                    }

    //4 Disegnare i cerchi sull'immagine originale in base ai risultati della Trasformata di Hough per i cerchi
    dst=src.clone();
    for(int r=Rmin; r<Rmax; r++)
        for(int b=0; b<edges.rows; b++)
            for(int a=0; a<edges.cols; a++)
                if(votes[b][a][r-Rmin] > HoughTH){
                    circle(dst, Point(a,b), 3, Scalar(0), 2, 8, 0);
                    circle(dst, Point(a,b), r, Scalar(0), 2, 8, 0);
                }
}

int main(int argc, char** argv){

    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if(src.empty()) return -1;

    Mat dst;
    int cannyLTH = 150; // Limite inferiore per la rilevazione di bordi tramite l'algoritmo di Canny.
    int cannyHTH = 230; // Limite superiore per la rilevazione di bordi tramite l'algoritmo di Canny.
    int HoughTH = 130; // Soglia per determinare se un determinato punto e raggio hanno abbastanza voti nella trasformata di Hough per essere considerati un cerchio rilevato.
    int Rmin = 40, Rmax = 130; // Specificano i raggi minimi e massimi dei cerchi che il programma cercherà di rilevare utilizzando la trasformata di Hough.
    houghCircles(src,dst,cannyLTH,cannyHTH,HoughTH,Rmin,Rmax);

    imshow("src",src);
    imshow("dst",dst);
    waitKey(0);
    return 0;
}
