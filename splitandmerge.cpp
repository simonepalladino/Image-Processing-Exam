#include <stdlib.h>
#include <opencv2/opencv.hpp>

// Dichiarazioni globali di variabili
float tSize;
int th;

using namespace std;
using namespace cv;

// Definizione della classe TNode che rappresenta un nodo dell'albero
class TNode {
public:
    Rect region;
    TNode *UL, *UR, *LL, *LR;
    vector<TNode*> merged;
    vector<bool> mergedB = vector<bool>(4, false);
    double stddev, mean;

    // Costruttore della classe
    TNode(Rect R) { region = R; UL = UR = LL = LR = nullptr; }

    // Aggiunge una regione al vettore delle regioni unite
    void addRegion(TNode* R) { merged.push_back(R); }

    // Imposta il flag mergedB per una regione specifica
    void setMergedB(int i) { mergedB[i] = true; }
};

// Funzione ricorsiva per la divisione dell'immagine in nodi dell'albero
TNode* split(Mat& src, Rect R) {

    // Creazione del nodo radice
    TNode* root = new TNode(R);

    // Calcolo della deviazione standard e della media della regione
    Scalar stddev, mean;
    meanStdDev(src(R), mean, stddev);
    root->stddev = stddev[0];
    root->mean = mean[0];

    // Se la regione supera le soglie, si procede con la divisione
    if (R.width > tSize && root->stddev > th) {

        // Definizione delle quattro sotto-regioni
        Rect ul(R.x, R.y, R.height / 2, R.width / 2);
        root->UL = split(src, ul);

        Rect ur(R.x, R.y + R.width / 2, R.height / 2, R.width / 2);
        root->UR = split(src, ur);

        Rect ll(R.x + R.height / 2, R.y, R.height / 2, R.width / 2);
        root->LL = split(src, ll);

        Rect lr(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2);
        root->LR = split(src, lr);
    }

    // Disegna un rettangolo sulla regione corrente nell'immagine di input
    rectangle(src, R, Scalar(0));
    return root;
}

/*
 _____  _____
|UL|UR| |0|1|
------- -----
|LL|LR| |3|2|
*/

// Funzione ricorsiva per la fusione delle regioni dell'albero
void merge(TNode* root) {

    if (root->region.width > tSize && root->stddev > th) {
        // UL-UR
        if (root->UL->stddev <= th && root->UR->stddev <= th) {
            root->addRegion(root->UL); root->setMergedB(0);
            root->addRegion(root->UR); root->setMergedB(1);
            // LL-LR
            if (root->LL->stddev <= th && root->LR->stddev <= th) {
                root->addRegion(root->LL); root->setMergedB(3);
                root->addRegion(root->LR); root->setMergedB(2);
            } else {
                merge(root->LL);
                merge(root->LR);
            }
        }
        // UR-LR
        else if (root->UR->stddev <= th && root->LR->stddev <= th) {
            root->addRegion(root->UR); root->setMergedB(1);
            root->addRegion(root->LR); root->setMergedB(2);
            // UL-LL
            if (root->UL->stddev <= th && root->LL->stddev <= th) {
                root->addRegion(root->UL); root->setMergedB(0);
                root->addRegion(root->LL); root->setMergedB(3);
            } else {
                merge(root->UL);
                merge(root->LL);
            }
        }
        // LL-LR
        else if (root->LL->stddev <= th && root->LR->stddev <= th) {
            root->addRegion(root->LL); root->setMergedB(3);
            root->addRegion(root->LR); root->setMergedB(2);
            // UL-UR
            if (root->UL->stddev <= th && root->UR->stddev <= th) {
                root->addRegion(root->UL); root->setMergedB(0);
                root->addRegion(root->UR); root->setMergedB(1);
            } else {
                merge(root->UL);
                merge(root->UR);
            }
        }
        // UL-LL
        else if (root->UL->stddev <= th && root->LL->stddev <= th) {
            root->addRegion(root->UL); root->setMergedB(0);
            root->addRegion(root->LL); root->setMergedB(3);
            // UR-LR
            if (root->UR->stddev <= th && root->LR->stddev <= th) {
                root->addRegion(root->UR); root->setMergedB(1);
                root->addRegion(root->LR); root->setMergedB(2);
            } else {
                merge(root->UR);
                merge(root->LR);
            }
        } else {
            // Altrimenti, si procede con la fusione in tutte le direzioni
            merge(root->UL);
            merge(root->UR);
            merge(root->LL);
            merge(root->LR);
        }
    } else {
        // Se la regione non supera le soglie, si aggiunge la regione corrente al vettore delle regioni unite
        root->addRegion(root);
        // Imposta tutti i flag mergedB a true indicando che la regione corrente è stata fusa in tutte le direzioni
        for (int i = 0; i < 4; i++) root->setMergedB(i);
    }
}

// Funzione per applicare la segmentazione all'immagine di input
void segment(TNode* root, Mat& src) {

    // Ottenere le regioni unite dalla radice
    vector<TNode*> tmp = root->merged;

    if (tmp.size() == 0) {
        // Se la radice non ha regioni unite, si procede con la segmentazione ricorsiva nei nodi figli
        segment(root->UL, src);
        segment(root->UR, src);
        segment(root->LR, src);
        segment(root->LL, src);
    } else {
        // Se

 la radice ha regioni unite
        double val = 0;
        for (auto x : tmp)
            val += x->mean;
        val /= tmp.size();

        // Assegna il valore medio alle regioni unite nell'immagine di output
        for (auto x : tmp)
            src(x->region) = (int)val;

        // Se ci sono più di una regione unita, si procede con la segmentazione ricorsiva nei nodi figli
        if (tmp.size() > 1) {
            if (!root->mergedB[0])
                segment(root->UL, src);
            if (!root->mergedB[1])
                segment(root->UR, src);
            if (!root->mergedB[2])
                segment(root->LR, src);
            if (!root->mergedB[3])
                segment(root->LL, src);
        }
    }
}

// Funzione principale per eseguire la divisione e la fusione dell'immagine
void splitAndMerge(Mat& src, Mat& segmented) {

    // Calcola la dimensione della potenza di due più vicina alla dimensione dell'immagine
    int exponent = log(min(src.rows, src.cols)) / log(2);
    int s = pow(2.0, double(exponent));

    // Esegue la copia dell'immagine e applica una sfocatura gaussiana
    Mat croppedSrc = src(Rect(0, 0, s, s)).clone();
    GaussianBlur(croppedSrc, croppedSrc, Size(3, 3), 0, 0);

    // Chiama la funzione split per ottenere l'albero di segmentazione
    TNode* root = split(croppedSrc, Rect(0, 0, croppedSrc.rows, croppedSrc.cols));
    
    // Chiama la funzione merge per eseguire la fusione delle regioni
    merge(root);

    // Inizializza l'immagine di output con la dimensione della potenza di due più vicina
    segmented = src(Rect(0, 0, s, s)).clone();

    // Chiama la funzione segment per assegnare i valori medi alle regioni unite
    segment(root, segmented);

    // Mostra le immagini di input e output
    imshow("croppedSrc", croppedSrc);
    imshow("segmented", segmented);
    waitKey(0);
}

int main(int argc, char** argv) {

    // Legge l'immagine di input in scala di grigi
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    // Imposta le soglie per la divisione e la fusione
    tSize = 8;
    th = 6;

    // Inizializza l'immagine di output
    Mat segmented;
    
    // Chiama la funzione principale per eseguire la divisione e la fusione
    splitAndMerge(src, segmented);

    return 0;
}
```