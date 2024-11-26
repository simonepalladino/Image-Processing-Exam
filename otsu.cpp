#include <opencv2/opencv.hpp>
#include <stdlib.h>

/*
L'algoritmo di Otsu è un metodo di binarizzazione automatica che determina una soglia ottimale per 
separare un'immagine in due classi (solitamente, oggetti e sfondo) in modo che la varianza tra 
le due classi sia massima. La varianza massima indica che le due classi sono il più possibile differenti.
*/

using namespace cv;
using namespace std;

// Funzione per calcolare un istogramma normalizzato per un'immagine in scala di grigi
vector<double> normalizedHistogram(Mat& src)
{
    // Vettore per memorizzare l'istogramma normalizzato con 256 bin (uno per ogni intensità)
    vector<double> his(256,0);
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


int otsu(Mat& src)
{
    vector<double> his = normalizedHistogram(src);
    // Inizializza la variabile per la media ponderata globale (gMean) a 0.0
    double gMean = 0.0f;
    // Calcola la media ponderata globale sommando la moltiplicazione tra l'intensità i e la sua frequenza normalizzata (his[i])
    for (int i=0; i<256; i++)
    {
        gMean+= i*his[i];
    }
    // Inizializza variabili necessarie per il calcolo della soglia ottimale
    double currProb1 = 0.0f; // Probabilità cumulativa della classe 1
    double currCumMean = 0.0f; // Media cumulativa della classe 1
    double currIntVar = 0.0f; // Varianza interclasse corrente
    double maxVar = 0.0f; // Massima varianza interclasse trovata
    int kstar = 0; // Soglia ottimale

    // Loop attraverso i livelli di intensità
    for (int i=0; i<256; i++)
    {
        currProb1 += his[i];
        currCumMean += i*his[i];
        // Calcolo della varianza interclasse corrente utilizzando la formula di Otsu
        currIntVar = pow(gMean*currProb1-currCumMean,2)/(currProb1*(1-currProb1));
        //Se la varianza interclasse corrente è maggiore della massima trovata finora,
        //aggiorna la massima varianza e registra la soglia corrente come ottimale (kstar)
        if(currIntVar > maxVar)
        {
            maxVar = currIntVar;
            kstar = i;
        }
    }
    return kstar;
    
}


int main(int argc, char**argv)
{
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty())
        return -1;
    Mat dst, gsrc;
    //Viene applicato un filtro di sfocatura gaussiana alla copia dell'immagine gsrc
    GaussianBlur(src,gsrc,Size(3,3),0,0);
    //calcolo la soglia con otsu
    int th = otsu(src);

    //L'immagine di input (src) viene binarizzata utilizzando la soglia ottimale calcolata. 
    //I pixel sopra la soglia diventano bianchi, mentre quelli sotto diventano neri.
    threshold(src,dst,th,255,THRESH_BINARY);

    imshow("src",src);
    imshow("dst",dst);
    waitKey(0);
    return 0;
}
