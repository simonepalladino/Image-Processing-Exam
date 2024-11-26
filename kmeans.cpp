#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <time.h>

using namespace std;
using namespace cv;

// Funzione per l'algoritmo di clustering k-means su immagini in scala di grigi
void kmeans(Mat& src, Mat& dst, int k){

    // Inizializza il generatore di numeri casuali con il tempo corrente
    srand(time(NULL));

    // 1. Inizializza i centroidi in posizioni casuali dell'immagine
    vector<uchar> c(k,0);
    for(int i=0; i<k; i++){
        int randRow = rand()%src.rows;
        int randCol = rand()%src.cols;
        c[i] = src.at<uchar>(randRow,randCol);
    }

    // 2 e 3. Esegui l'algoritmo k-means
    double epsilon = 0.01f; // rappresenta il criterio di convergenza. L'algoritmo continua a iterare finché la variazione tra i centroidi in iterazioni successive è superiore a epsilon. 
    bool is_c_varied = true; // è una variabile booleana utilizzata per verificare se i centroidi cambiano durante l'iterazione corrente dell'algoritmo.
    int maxIterations = 50; //rappresenta il numero massimo di iterazioni che l'algoritmo eseguirà prima di fermarsi, indipendentemente dalla convergenza.
    int it; // variabile che tiene traccia del numero corrente di iterazioni.
    //oldmean e newmean sono vettori che contengono la media dei pixel in ciascun cluster nelle iterazioni corrente e precedente, rispettivamente.    
    vector<double> oldmean(k,0.0f); 
    vector<double> newmean(k,0.0f);
    //cluster è un vettore di vettori di Point. Ogni vettore interno rappresenta un cluster e contiene le coordinate dei pixel assegnati a quel cluster durante l'iterazione corrente.
    vector<vector<Point>> cluster(k);

    vector<uchar> dist(k,0); //dist è un vettore che viene utilizzato per memorizzare le distanze tra ciascun pixel e i centroidi dei cluster correnti. 
    int minDistIdx; //minDistIdx è un indice che verrà utilizzato per memorizzare l'indice del cluster con la distanza minima da un determinato pixel.

    // Il ciclo while si ripete fino a quando i centroidi variano significativamente (is_c_varied) o fino a quando viene raggiunto il numero massimo di iterazioni (maxIterations).
    while(is_c_varied && it++ < maxIterations){
    
        // Resetta i cluster
        is_c_varied = false;
        for(int i=0; i<k; i++) cluster[i].clear(); //Resetta i cluster, svuotando i vettori di punti associati a ciascun cluster.
        for(int i=0; i<k; i++) oldmean[i] = newmean[i]; //Salva le medie dei cluster dell'iterazione precedente in oldmean.

        // Scorro i pixel dell'immagine
        for(int x=0; x<src.rows; x++){
            for(int y=0; y<src.cols; y++){
                //calcola la distanza di ciascun pixel dai centroidi dei cluster correnti e memorizza i risultati in dist.
                for(int i=0; i<k; i++){
                    dist[i] = abs(c[i] - src.at<uchar>(x,y));
                }
                minDistIdx = min_element(dist.begin(),dist.end())-dist.begin(); //viene poi calcolato come l'indice del cluster con la distanza minima da un determinato pixel, utilizzando min_element.
                cluster[minDistIdx].push_back(Point(x,y)); // punti vengono assegnati ai cluster in base alla distanza minima calcolata, e i risultati vengono memorizzati nei vettori di cluster.
            }
        }

        // Calcola i nuovi centroidi
        for(int i=0; i<k; i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0; j<csize; j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                newmean[i] += src.at<uchar>(cx,cy);
            }
            newmean[i] /= csize;
            c[i] = uchar(newmean[i]);
        }

        // Controlla se i centroidi sono variati
        for(int i=0; i<k; i++)
            if( !(abs(newmean[i]-oldmean[i]) <= epsilon) )
                is_c_varied = true;

        cout<<it<<endl;
    }

    // 4. Sostituisci i pixel nell'immagine con i centroidi dei rispettivi cluster
    dst = src.clone();
    for(int i=0; i<k; i++){
        int csize = static_cast<int>(cluster[i].size());
        for(int j=0; j<csize; j++){
            int cx = cluster[i][j].x;
            int cy = cluster[i][j].y;
            dst.at<uchar>(cx,cy) = c[i];
        }
    }

}

// Funzione per l'algoritmo di clustering k-means su immagini RGB
void kmeansRGB(Mat& src, Mat& dst, int k){

    // Inizializza il generatore di numeri casuali con il tempo corrente
    srand(time(NULL));

    // 1. Inizializza i centroidi in posizioni casuali dell'immagine RGB
    vector<Vec3b> c(k,0);
    for(int i=0; i<k; i++){
        int randRow = rand()%src.rows;
        int randCol = rand()%src.cols;
        c[i] = src.at<Vec3b>(randRow,randCol);
    }

    // 2 e 3. Esegui l'algoritmo k-means
    double epsilon = 0.01f;
    bool is_c_varied = true;
    int maxIterations = 50;
    int it;
    vector<Vec3d> oldmean(k,0.0f);
    vector<Vec3d> newmean(k,0.0f);
    vector<vector<Point>> cluster(k);

    // Split dei canali RGB dell'immagine
    double diffBlue, diffGreen, diffRed;
    vector<uchar> dist(k,0);
    int minDistIdx;

    vector<Mat> srcChannels(3);
    split(src, srcChannels);

    while(is_c_varied && it++ < maxIterations){
        
        // Resetta i cluster
        is_c_varied = false;
        for(int i=0; i<k; i++) cluster[i].clear();
        for(int i=0; i<k; i++) oldmean[i] = newmean[i];

        // Assegna ogni punto al cluster più vicino
        for(int x=0; x<src.rows; x++){
            for(int y=0; y<src.cols; y++){
                for(int i=0; i<k; i++){
                    diffBlue  = c[i].val[0] - srcChannels[0].at<uchar>(x,y);
                    diffGreen = c[i].val[1] - srcChannels[1].at<uchar>(x,y);
                    diffRed   = c[i].val[2] - srcChannels[2].at<uchar>(x,y);
                    dist[i] = sqrt(pow(diffBlue, 2) + pow(diffGreen,2) + pow(diffRed,2));
                }
                minDistIdx = min_element(dist.begin(),dist.end())-dist.begin();
                cluster[minDistIdx].push_back(Point(x,y));
            }
        }

        // Calcola i nuovi centroidi
        for(int i=0; i<k; i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0; j<csize; j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                newmean[i].val[0] += srcChannels[0].at<uchar>(cx,cy);
                newmean[i].val[1] += srcChannels[1].at<uchar>(cx,cy);
                newmean[i].val[2] += srcChannels[2].at<uchar>(cx,cy);
            }
            newmean[i] /= csize;
            c[i] = newmean[i];
        }

        // Controlla se i centroidi sono variati
        double val = 0.0f;
        for(int i=0; i<k; i++){
            for(int ch=0; ch<3; ch++)
                val += newmean[i].val[ch]-oldmean[i].val[ch];
            val /= 3;
            if( abs(val) <= epsilon)
                is_c_varied = true;
        }

        cout<<it<<endl;
    }

    // 4. Sostituisci i pixel nell'immagine con i centroidi dei rispettivi cluster
    dst = src.clone();
    for(int i=0; i<k; i++){
        int csize = static_cast<int>(cluster[i].size());
        for(int j=0; j<csize; j++){
            int cx = cluster[i][j].x;
            int cy = cluster[i][j].y;
            dst.at<Vec3b>(cx,cy) = c[i];
        }
    }

}

int main(int argc, char** argv)
{
    Mat src = imread(argv[1]);
    if(src.empty()) return -1;

    // Converti l'immagine a colori in scala di grigi
    Mat greySrc;
    cvtColor(src, greySrc, COLOR_BGR2GRAY);

    // Applica l'algoritmo k-means all'immagine in scala di grigi
    Mat dst;
    kmeans(greySrc,dst,2);

    // Applica l'algoritmo k-means all'immagine RGB
    Mat dstColor;
    kmeansRGB(src,dstColor,5);

    // Mostra le immagini originali e quelle elaborate
    imshow("src",src);
    imshow("kmeans",dst);
    imshow("kmeansRGB",dstColor);

    waitKey(0);
    return 0;
}









