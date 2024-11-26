#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;


void circleCorners(Mat& src, Mat& dst, int th)
{
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            //Questo blocco di codice implementa la logica per 
            //disegnare cerchi nei punti chiave individuati sulla matrice di destinazione dst
            if((int)src.at<float>(i,j)>th)
                /*
                dst: Questo è il primo parametro e rappresenta la matrice su cui verrà disegnato il cerchio. 
                Nel contesto del tuo codice, dst è l'immagine di destinazione dove vengono evidenziati i punti chiave.
                Point(j, i): Il secondo parametro rappresenta il centro del cerchio e viene specificato 
                come un oggetto Point. j è l'indice di colonna (coordinata x) e i è l'indice di riga (coordinata y) nell'immagine. 
                Questo indica la posizione in cui il cerchio verrà disegnato.

                5: Il terzo parametro è il raggio del cerchio. In questo caso, è impostato su 5, quindi il cerchio avrà un raggio di 5 pixel.

                Scalar(0): Il quarto parametro rappresenta il colore del cerchio. Scalar(0) indica che il colore è nero. 
                Nel contesto di immagini in scala di grigi, dove ogni pixel può avere un valore tra 0 e 255, un colore nero è rappresentato da Scalar(0).

                1: Il quinto parametro è lo spessore del contorno del cerchio. In questo caso, è impostato su 1, quindi il contorno del cerchio sarà sottile.
                */
                circle(dst,Point(j,i),5,Scalar(0),1);
        }
    }
}



void harris(Mat& src, Mat &dst, int ksize, int th, float k)
{
    Mat Dx, Dy;
	Sobel(src,Dx,CV_32FC1,1,0,ksize);
	Sobel(src,Dy,CV_32FC1,0,1,ksize);

    Mat Dx2, Dy2, DxDy;
    pow(Dx,2,Dx2);
	pow(Dy,2,Dy2);

    multiply(Dx,Dy,DxDy);
    
    Mat C00, C01, C10, C11;

    /*
    GaussianBlur(Dx2, C00, Size(7,7), 2,0); 
    Applica il filtro di Gauss alla matrice Dx2 (quadrato della derivata rispetto a x) 
    utilizzando un   kernel di dimensione 7x7, con una deviazione standard di 2 lungo 
    la direzione x. Il risultato viene memorizzato nella matrice C00.
    */
    GaussianBlur(Dx2,C00,Size(7,7),2,0);
	GaussianBlur(Dy2,C11,Size(7,7),0,2);
	GaussianBlur(DxDy,C01,Size(7,7),2,2);
    
    /*
    Assegna il valore della matrice C01 alla matrice C10. Questo è possibile 
    perché la matrice C01 è simmetrica rispetto alla trasposizione, 
    e quindi C10 può essere considerata uguale a C01 nel contesto di questo algoritmo.
    */
    C10 = C01;

    //Questa parte del codice calcola la metrica di Harris per ogni pixel dell'immagine.
    
    Mat det, trace, trace2, R, PPD, PSD;

    multiply(C00,C11,PPD);
    multiply(C01,C10,PSD);

    det = PPD-PSD; //Calcola il determinante della matrice di struttura H, 
                     //dove H è una matrice che   rappresenta le derivate seconde dell'immagine.

    trace = C00+C11; //Calcola la traccia della matrice di struttura H. 
                       //La traccia è la somma degli elementi diagonali della matrice.
	pow(trace,2,trace2);
    
    /*
    Calcola la metrica di Harris utilizzando il determinante, la traccia e il parametro k. 
    La formula è R = det(H) - k * (trace(H))^2. Questa metrica è utilizzata per identificare 
    punti chiave nell'immagine. Valori elevati di R indicano regioni con forti cambiamenti 
    di intensità, che sono spesso correlate a bordi o caratteristiche importanti.
    */
    R = det-k*trace2;

    //rappresenta la magnitudine dei gradienti
    //primo parametro R rappresenta la matrice di input
    //secondo parametro R rappresenta la matrice di output
    //terzo parametro 0 valore minimo dopo la normalizzazione 
    //quarto parametro 255 valore massimo dopo la normalizzazione 
    //quinto parametro tipo di normalizzazione
    //sesto parametro tipo della matrice di output
    normalize(R,R,0,255,NORM_MINMAX,CV_32FC1);

    /*
    Converte la matrice normalizzata R in un tipo di dato senza segno a 8 bit (CV_8U). 
    Questa operazione è importante per preparare i dati per la visualizzazione, poiché la 
    maggior parte delle immagini in OpenCV sono di tipo CV_8U. La funzione convertScaleAbs 
    converte i dati a virgola mobile in interi senza segno e scala i valori in modo che 
    rimangano nell'intervallo [0, 255].
    */
    convertScaleAbs(R,dst);

    /*
    Questa funzione viene chiamata per evidenziare i punti chiave sull'immagine di destinazione dst. 
    La funzione circleCorners prende in input la matrice normalizzata R e disegna dei cerchi nei 
    punti dove i valori superano una certa soglia th.
    */
    circleCorners(R,dst,th);
}

int main(int argc, char**argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

	Mat dst;
    /*
ksize: Questo parametro indica la dimensione 
della finestra del filtro Sobel utilizzato per calcolare le derivate parziali. 
In questo caso, è impostato su 3, 
il che significa che il filtro Sobel sarà una matrice 3x3.        
Dimensioni più grandi della finestra 
possono essere utilizzate per considerare aree 
più grandi dell'immagine per il calcolo delle derivate.
    
th: Questo parametro rappresenta una soglia. I punti nell'immagine con una risposta di Harris   superiore a questa soglia verranno considerati punti chiave. In altre parole, è un valore di soglia per decidere quali regioni sono abbastanza "interessanti" per essere considerate punti chiave.
    
k: Questo è il parametro k utilizzato nella formula di calcolo della metrica di Harris. La formula di Harris include una costante k che regola la sensibilità dell'algoritmo alla differenza tra la risposta di Harris di un punto e la sua risposta media. Valori tipici per k sono compresi tra 0.04 e 0.15. Un valore più alto di k rende l'algoritmo più sensibile alle variazioni locali di intensità, evidenziando angoli più nitidi.
    */
    int ksize = 3, th = 150;
	float k = 0.117;
	harris(src,dst,ksize,th,k);

	imshow("src",src);
	imshow("dst",dst);
	waitKey(0);

	return 0;
}
