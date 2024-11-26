#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

void hysteresisThreshold(Mat& nms, Mat& dst, int lth, int hth){
    for(int i=1; i<nms.rows-1; i++){
		for(int j=1; j<nms.cols-1; j++){
            /*
            L'istruzione if(nms.at<uchar>(i,j) > hth) controlla se il valore del pixel nella 
            matrice nms alla posizione (i, j) è maggiore della soglia alta (hth). 
            In termini più semplici, sta verificando se la magnitudine del 
            gradiente in quel punto è sopra una determinata soglia.
            */
            if(nms.at<uchar>(i,j)>hth){
				dst.at<uchar>(i,j)=255;
               
                for(int k=-1; k<=1; k++)
					for(int l=-1; l<=1; l++)
                            /*
                            Questa parte del codice è una condizione all'interno di 
                            un doppio ciclo for che controlla i pixel vicini al pixel corrente nelle direzioni orizzontali e        
                            verticali (usando k e l come offset rispettivamente).
                            L'if verifica due condizioni:

                            nms.at<uchar>(i+k, j+l) > lth: Controlla se la magnitudine del gradiente del pixel vicino è superiore alla soglia bassa (lth).
                            nms.at<uchar>(i+k, j+l) < hth: Controlla se la magnitudine del gradiente del pixel vicino è inferiore alla soglia alta (hth).
                            */            
						    if(nms.at<uchar>(i+k,j+l) > lth && nms.at<uchar>(i+k,j+l) < hth)
                                /*
                                Questa istruzione assegna il valore 255 al pixel nella matrice dst alla posizione (i+k, j+l).
                                In termini dell'algoritmo di Canny, questo di solito significa che il pixel è collegato a un
                                bordo ritenuto forte durante la fase di isteresi. Quindi, se il pixel corrente ha una magnitudine 
                                del gradiente superiore alla soglia alta e il suo vicino ha una magnitudine del gradiente tra le soglie bassa e alta, 
                                allora il pixel corrente viene considerato parte del bordo e viene impostato a 255.
                                */
							    dst.at<uchar>(i+k,j+l) = 255;
            }else{
				dst.at<uchar>(i,j)=0;
			}
		}
	}
}



void nonMaximaSuppression(Mat& mag, Mat& orientations, Mat& nms)
{
    float angle;
	for(int i=1; i<mag.rows-1; i++){
		for(int j=1; j<mag.cols-1; j++){
            /*.at<float>(i,j) indica che il tipo di dati a cui stai accedendo è float, questo è un tipo di dato della matrice orientations e ci 
            permette di accedere alla posizione i,j */
            angle = orientations.at<float>(i,j);

            /* OPERATORE TERNARIO
            angle > 180: È la condizione. Se l'angolo è maggiore di 180 gradi.
            angle - 360: È il valore restituito se la condizione è vera. Sottrae 360 gradi dall'angolo.
            angle: È il valore restituito se la condizione è falsa. In tal caso, l'angolo rimane invariato.
            */
            angle = angle > 180 ? angle - 360 : angle;
            if((-22.5 < angle && angle <= 22.5) || (157.5 < angle && angle <= -157.5)){ //horizontal
				if(mag.at<uchar>(i,j) > mag.at<uchar>(i,j-1) && mag.at<uchar>(i,j) > mag.at<uchar>(i,j+1)){
					nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
				}
			}
			else if((-67.5 < angle && angle <= -22.5) || (112.5 < angle && angle <= 157.5)){ //+45
				if(mag.at<uchar>(i,j) > mag.at<uchar>(i-1,j+1) && mag.at<uchar>(i,j) > mag.at<uchar>(i+1,j-1)){
					nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
				}
			}
			else if((-112.5 < angle && angle <= -67.5) || (67.5 < angle && angle <= 112.5)){ //vertical
				if(mag.at<uchar>(i,j) > mag.at<uchar>(i-1,j) && mag.at<uchar>(i,j) > mag.at<uchar>(i+1,j)){
					nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
				}
			}
			else if((-157.5 < angle && angle <= -112.5) || (67.5 < angle && angle <= 22.5)){ //-45
				if(mag.at<uchar>(i,j) > mag.at<uchar>(i-1,j-1) && mag.at<uchar>(i,j) > mag.at<uchar>(i+1,j+1)){
					nms.at<uchar>(i,j) = mag.at<uchar>(i,j);
				}
			}
		}
	}
}



void canny(Mat& src, Mat& dst, int lth, int hth, int ksize)
{
    Mat gauss;
    /*
    primo parametro src: L'immagine di input originale.
    
    secondo parametro gauss: La matrice in cui verrà memorizzata l'immagine sfocata.
    
    terzo parametro Size(3, 3): La dimensione del kernel gaussiano. In questo caso, è una matrice 3x3, 
    che determina la dimensione della finestra di sfocatura.
    
    quarto parametro 0: Deviazione standard lungo l'asse X del kernel gaussiano. Un valore di 0 consente a 
    OpenCV di calcolare automaticamente la deviazione standard in base alla dimensione del kernel.
    
    quinto parametro 0: Deviazione standard lungo l'asse Y del kernel gaussiano, con la stessa logica del parametro precedente. */

    GaussianBlur(src,gauss, Size(3,3),0,0); //sfocatura con la gaussiana dell'immagine di input

    Mat Dx, Dy, Dx2, Dy2, mag, orientations;
    
    //CV_32FC1 è il tipo di dato della matrice e in questo caso sono 32 bit float con C1 che rappresenta un solo canale
    Sobel(gauss, Dx, CV_32FC1, 1, 0, ksize);
	Sobel(gauss, Dy, CV_32FC1, 0, 1, ksize);
    pow(Dx,2,Dx2); //calcolo il quadrato della derivata dx e lo inserisco in dx2
    pow(Dy,2,Dy2); //calcolo il quadrato della derivata dy e lo inserisco in dy2
    sqrt(Dx2+Dy2,mag); //la radice della somma delle due derivate dx e dy e la inserisco nella matrice mag

    //rappresenta la magnitudine dei gradienti
    //primo parametro mag rappresenta la matrice di input
    //secondo parametro mag rappresenta la matrice di output
    //terzo parametro 0 valore minimo dopo la normalizzazione 
    //quarto parametro 255 valore massimo dopo la normalizzazione 
    //quinto parametro tipo di normalizzazione
    //sesto parametro tipo della matrice di output
    normalize(mag,mag,0,255,NORM_MINMAX,CV_8UC1);

    /*
    Questa istruzione calcola la fase (o direzione) dei gradienti utilizzando le derivate parziali Dx e Dy
    primo parametro: matrice delle derivate parziali rispetto a x
    secondo parametro: matrice delle derivate parziali rispetto a y
    terzo parametro: matrice che conterrà la direzione dei gradienti
    quarto parametro: Se impostato su true, la funzione restituirà le fasi (direzioni)in radianti; se impostato su false, restituirà le fasi in gradi.
    */
    phase(Dx,Dy,orientations,true);

    //creo una matrice di zeri di nome nms
    Mat nms = Mat::zeros(src.rows,src.cols,CV_8UC1);
    //richiamo della funzione nonMaximaSuppression
    /*
    La funzione eseguirà la soppressione dei non massimi, 
    assegnando i valori della magnitudine dei gradienti solo ai pixel che sono considerati 
    massimi lungo le direzioni specificate dalle fasi dei gradienti.
    */
    nonMaximaSuppression(mag,orientations,nms);

    dst = Mat::zeros(src.rows,src.cols,CV_8UC1);

    /*
    La funzione applica una soglia di isteresi all'immagine, collegando i pixel 
    che superano la soglia alta e sono considerati forti bordi, e propagando il 
    collegamento ai pixel sopra la soglia bassa che sono connessi a quelli sopra la soglia alta. Il risultato finale è memorizzato nella matrice dst.
    */
    hysteresisThreshold(nms, dst, lth, hth);
    
}


int main(int argc, char**argv)
{
    Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

	Mat dst;
	int lth = 20, hth = 80, ksize = 3; //lth soglia basse per la fase di isteresi, hth soglia alta per 
                                       //la fase di isteresi, ksize è la dimensione del kernel per i filtri di sobel

    canny(src,dst,lth,hth,ksize);

	imshow("src",src);
	imshow("dst",dst);
	waitKey(0);

	return 0;
    
}

