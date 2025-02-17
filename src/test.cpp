#include "conv.h"

int main(){

    int x = 6;
    int f = 3;

    int *X = cnn::createMatrix(x, x, static_cast<int>(-1));
    int *F = cnn::createMatrix(f, f, static_cast<int>(1));

    std::cout<<"X = "<<std::endl;
    cnn::displayMatrix(X, x, x);
    std::cout<<"F = "<<std::endl;
    cnn::displayMatrix(F, f, f);


    char mode = 'v';

    int *Y = cnn::convolution(X, F, x, x, f, f, mode);

    std::cout<<"Y = "<<std::endl;
    
    if(mode=='s') cnn::displayMatrix(Y, x, x);
    else if(mode=='v') cnn::displayMatrix(Y, x-f+1, x-f+1);
    
    free(X);
    free(F);
    free(Y);
    
    return 0;
}