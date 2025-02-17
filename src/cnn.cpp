#include<iostream>
#include<functional>
#include <iomanip>
#include <random>

template <typename T>
T* createMatrix(int x0, int x1, T fillval=0){
    T *Y = new T[x0*x1];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1,10);

    for(int i=0; i<x0*x1; i++){
        if(fillval==-1){
            int val = dis(gen);
            Y[i] = static_cast<T>(val);
        }
        else{
            Y[i] = static_cast<T>(fillval);
        }
        
    }   
    return Y;
}


template <typename T>
void displayMatrix(T*X, int x0, int x1){

    for(int i=0; i<x0; i++){
        for(int j=0; j<x1; j++){
            std::cout<< std::setw(3) <<X[i*x1 + j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

template <typename T>
T* convSame(T* X, T* F, int x0, int x1, int f0, int f1){
    T fillval = static_cast<T>(0);
    T* Y = createMatrix(x0, x1, fillval);

    for(int i=0; i<x0; i++){
        for(int j=0; j<x1; j++){
            for(int k0=0; k0<f0; k0++){
                for(int k1=0; k1<f1; k1++){
                    int xr = i - static_cast<int>((f0-1)/2);
                    int xc = j - static_cast<int>((f1-1)/2);
                    if(xr+k0>=0 && xr+k0<x0 && xc+k1>=0 && xc+k1<x1){
                        Y[i*x1 + j] += X[(xr+k0)*x1 + xc+k1]*F[k0*f1 + k1];
                    }
                }
            }
        }
    }
    return Y;  
}

template <typename T>
T* convValid(T* X, T* F, int x0, int x1, int f0, int f1){
    
    int y0 = x0-f0+1;
    int y1 = x1-f1+1;
    T fillval = static_cast<T>(0);
    T* Y = createMatrix(y0, y1, fillval);

    for(int i=0; i<y0; i++){
        for(int j=0; j<y1; j++){
            for(int k0=0; k0<f0; k0++){
                for(int k1=0; k1<f1; k1++){
                    int xr = i;
                    int xc = j;
                    Y[i*y1 + j] += X[(xr+k0)*x1 + xc+k1]*F[k0*f1 + k1];
                }
            }
        }
    }

    return Y;   

}

template <typename T>
T* convolution(T* X, T* F, int x0, int x1, int f0, int f1, char mode='s'){

    T* (*convFunc)(T*, T*, int, int, int, int) = convSame;

    if (mode=='s')
    {
        convFunc = static_cast<T* (*)(T*, T*, int, int, int, int)>(convSame<T>); 

    }
    
    else if(mode=='v') 
    {

        convFunc = static_cast<T* (*)(T*, T*, int, int, int, int)>(convValid<T>);  
    }
    
    else {

        std::cout<<"Input convolution mode is not supported! (Please enter s (same) or v(valid))\n";
        return static_cast<T*>(nullptr);
    }

    T* Y = convFunc(X, F, x0, x1, f0, f1);
    return Y;
}



int main(){

    int x = 6;
    int f = 3;

    int *X = createMatrix(x, x, static_cast<int>(-1));
    int *F = createMatrix(f, f, static_cast<int>(1));

    std::cout<<"X = "<<std::endl;
    displayMatrix(X, x, x);
    std::cout<<"F = "<<std::endl;
    displayMatrix(F, f, f);


    char mode = 'v';

    int *Y = convolution(X, F, x, x, f, f, mode);

    std::cout<<"Y = "<<std::endl;
    
    if(mode=='s') displayMatrix(Y, x, x);
    else if(mode=='v') displayMatrix(Y, x-f+1, x-f+1);
    
    free(X);
    free(F);
    free(Y);
    
    return 0;
}

