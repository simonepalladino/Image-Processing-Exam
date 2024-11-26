#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Funzione per calcolare un istogramma normalizzato per un'immagine in scala di grigi 
vector<double> normalizedHistogram(Mat& src){
    // Vettore per memorizzare l'istogramma normalizzato con 256 bin (uno per ogni intensità)
    vector<double> his(256,0.0f);
    // Scorrimento di tutti i pixel dell'immagine
    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++)
            // Incremento del contatore relativo all'intensità del pixel corrente
            his[src.at<uchar>(i,j)]++;
    // Normalizzazione dell'istogramma dividendo ogni bin per il numero totale di pixel nell'immagine
    for(int i=0; i<256; i++)
        his[i] /= src.rows*src.cols;

    // Restituzione dell'istogramma normalizzato
    return his;
}

vector<int> otsu2k(Mat& src){

    vector<double> his = normalizedHistogram(src);

    // Inizializza la variabile per la media ponderata globale (gMean) a 0.0
    double gMean = 0.0f;
    // Calcola la media ponderata globale sommando la moltiplicazione tra l'intensità i e la sua frequenza normalizzata (his[i])
    for(int i=0; i<256; i++)
        gMean += i*his[i];

    /*
    Questa parte del codice implementa l'algoritmo Otsu per la ricerca di due soglie ottimali 
    in un'immagine in scala di grigi. L'obiettivo è dividere l'istogramma dell'immagine in
    tre classi, massimizzando la varianza tra le classi
    */
    vector<double> currProb(3,0.0f); //Un vettore di probabilità corrente per ciascuna classe.
    vector<double> currCumMean(3,0.0f); //Un vettore di medie cumulative correnti per ciascuna classe.
    double currIntVar = 0.0f; // La varianza tra le classi corrente.
    double maxVar = 0.0f; //La massima varianza tra le classi trovata durante la ricerca.
    vector<int> kstar(2,0); //Un vettore che conterrà le due soglie ottimali.
    for(int i=0; i<256-2; i++){
         // Itera attraverso tutte le possibili soglie per la prima classe
        currProb[0] += his[i];
        currCumMean[0] += i*his[i];
        for(int j=i+1; j<256-1; j++){
            // Itera attraverso tutte le possibili soglie per la seconda classe
            currProb[1] += his[j];
            currCumMean[1] += j*his[j];
            for(int k=j+1; k<256; k++){
                // Itera attraverso tutte le possibili soglie per la terza classe
                currProb[2] += his[k];
                currCumMean[2] += k*his[k];
                currIntVar = 0.0f;
                // Calcola la varianza tra le classi corrente usando le probabilità e le medie cumulative
                for(int w=0; w<3; w++)
                    currIntVar += currProb[w]*pow(currCumMean[w]/currProb[w]-gMean,2);
                // Verifica se la varianza corrente è maggiore della massima varianza trovata finora
                if(currIntVar > maxVar){
                    // Se sì, aggiorna la massima varianza e le soglie ottimali
                    maxVar = currIntVar;
                    kstar[0] = i;
                    kstar[1] = j;
                }
            }
            // Resetta le probabilità e le medie cumulative per la terza classe dopo ogni iterazione
            currProb[2] = currCumMean[2] = 0.0f;
        }
        // Resetta le probabilità e le medie cumulative per la seconda classe dopo ogni iterazione
        currProb[1] = currCumMean[1] = 0.0f;
    }

    //ritorno le soglie ottimali
    return kstar;
}

/*
La funzione multipleThresholds prende in input un'immagine in scala di grigi (src) 
e applica due soglie specificate (th1 e th2) per ottenere un'immagine binarizzata. 
La binarizzazione viene eseguita in modo che i pixel sopra la soglia th2 siano impostati a 255 (bianco), 
i pixel tra th1 e th2 siano impostati a 127 (grigio), e il resto dei pixel sia impostato a 0 (nero).
*/

void multipleThresholds(Mat& src, Mat& dst, int th1, int th2){

    // Stampa le soglie th1 e th2
    cout<<th1<<"\t"<<th2<<endl;
    // Inizializza l'immagine di destinazione con pixel neri
    dst = Mat::zeros(src.rows, src.cols, CV_8U);

    // Itera attraverso ogni pixel dell'immagine di input
    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++)
            // Applica la binarizzazione in base alle soglie
            if(src.at<uchar>(i,j) >= th2)
                dst.at<uchar>(i,j) = 255; // Pixel sopra th2 sono bianchi
            else if(src.at<uchar>(i,j) >= th1)
                dst.at<uchar>(i,j) = 127; // Pixel tra th1 e th2 sono grigi
}


int main(int argc, char** argv){

	Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

    Mat dst, gsrc;
    //Viene applicato un filtro di sfocatura gaussiana alla copia dell'immagine gsrc
    GaussianBlur(src,gsrc,Size(3,3),0,0);

    //calcolo le soglie multiple con otsu
    vector<int> ths = otsu2k(gsrc);
    multipleThresholds(gsrc,dst,ths[0],ths[1]);

    imshow("src",src);
    imshow("otsu2k",dst);
    waitKey(0);
	return 0;
}
