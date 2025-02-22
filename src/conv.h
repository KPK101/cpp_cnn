#ifndef CONV_H
#define CONV_H

#include<iostream>
#include<functional>
#include <iomanip>
#include <random>
#include<chrono>
#include <cassert>

#include<string>
namespace matrix{

    template <typename T>
    T* createMatrix(int x0, int x1, T fillval=0){
        // Create a matrix with value = fillvalue at each index
        T *Y = new T[x0*x1];
        
        // random seed to initalize random values
        std::random_device rd;
        auto seed = rd() ^ std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(1,10);
        
        for(int i=0; i<x0*x1; i++){
            // if fillval = -1 (random)
            if(fillval==-1){
                int val = dis(gen);
                Y[i] = static_cast<T>(val);
            }
            // else set value = fillval
            else{
                Y[i] = static_cast<T>(fillval);
            }
            
        }   
        return Y;
    }


    template <typename T>
    void displayMatrix(T*X, int x0, int x1){
        // utility function to display matrix
        for(int i=0; i<x0; i++){
            for(int j=0; j<x1; j++){
                std::cout<< std::setw(3) <<X[i*x1 + j]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n";
    }

}


namespace conv2D{    
    template <typename T>
    void convSame(T*Y, T* X, T* F, int x0, int x1, int f0, int f1){
        // Convolution - type "same"
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
    }

    template <typename T>
    void convValid(T*Y, T* X, T* F, int x0, int x1, int f0, int f1){
        // Convolution - type valid
        int y0 = x0-f0+1;
        int y1 = x1-f1+1;
        // T fillval = static_cast<T>(0);
        // T* Y = matrix::createMatrix(y0, y1, fillval);
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
    }

    template <typename T>
    void convolution(T*Y, T* X, T* F, int x0, int x1, int f0, int f1, char mode='s'){
        // 2D convolution function
        void (*convFunc)(T*, T*, T*, int, int, int, int) = convSame;
        if (mode=='s')
        {
            convFunc = static_cast<void (*)(T*, T*, T*, int, int, int, int)>(convSame<T>); 
        }
        else if(mode=='v') 
        {
            convFunc = static_cast<void (*)(T*, T*, T*, int, int, int, int)>(convValid<T>);  
        }
        else {
            std::cout<<"Input convolution mode is not supported! (Please enter s (same) or v(valid))\n";
            // return;
        }
        convFunc(Y, X, F, x0, x1, f0, f1);
    }   
    
}

namespace tensor{

    template<typename T>
    std::unique_ptr<T[]> createTensor(int N, int C, int H, int W, char state = 'c', int fillval=1){
        // create tensor of shape (N, C, H, W)
        size_t size = N*C*H*W;
        auto tensor = std::make_unique<T[]>(size);

        // Initialize with uniform random values
        std::random_device rd;
        std::mt19937 gen(rd());

        for(int i=0; i<N*C*H*W; i++){ 
            if(state=='r'){
                tensor[i] = std::uniform_real_distribution<T>(0,10)(gen);
            }else if (state=='z'){
                tensor[i] = 0;
            }else if(state=='f'){
                tensor[i] = fillval;
            }else{
                tensor[i] = 0;
                std::cout << "Invalid argument - set values to zero" << std::endl;
            }
        }
        return tensor;
    }
    

    // Template class for tensor
    template<typename T>
    class Tensor{
        public:
            // define tensor data elements
            int N, C, H, W;
            std::unique_ptr<T[]> data;
            
            // Default constructor
            Tensor() : N(0), C(0), H(0), data(nullptr) {}
                
            // Parametrized constructor
            Tensor(int n, int c, int h, int w, int state='z', int fillval=1) {
                // constructor
                N = n;
                C = c;
                H = h;
                W = w;
                data = createTensor<T>(N, C, H, W, state, fillval);
            }

            // Copy constructor
            Tensor(const Tensor &other) : N(other.N), C(other.C), H(other.H), W(other.W){

                size_t size = N*C*H*W;
                data = std::make_unique<T[]>(size);
                std::copy(other.data.get(), other.data.get()+size, data.get());
            }

            ~Tensor(){
                // destructor - delete dynamic tensor array 
                // no need to update as smart pointer was used!
            }

            // Copy assignment operator
            Tensor& operator=(const Tensor& other){
                if(this==&other) return *this;

                N = other.N;
                C = other.C;
                H = other.H;
                W = other.W;

                size_t size = N*C*H*W;
                data = std::make_unique<T[]>(size);
                std::copy(other.data.get(), other.data.get()+size, data.get());

                return *this;
                
            }

            // move constructor 
            Tensor(Tensor&& other) noexcept: N(other.N), C(other.C), H(other.H), W(other.W), data(std::move(other.data)){
                // std::cout << "Move constructor called!\n" ;
                other.N = 0;
                other.C = 0;
                other.H = 0;
                other.W = 0;
                other.data = nullptr;

            }

            // move assignment operator
            Tensor &operator=(Tensor&& other) noexcept{
                if(this!=&other){
                    N = other.N;
                    C = other.C;
                    H = other.H;
                    W = other.W;
                    data = std::move(other.data);

                    // cleanup
                    other.N = 0;
                    other.C = 0;
                    other.H = 0;
                    other.W = 0;
                    other.data = nullptr;
                }

                return *this;
            }


            T get(int n=0, int c=0, int h=0, int w=0){
                // get value at input coordinates (n,c,h,w)
                assert(n<N && c<C && h<H && w<W);
                return data[n*(C*H*W) + c*(H*W) + h*W + w];
            }
            
            T* getMatrix(int n, int c){
                // get cth channel of nth tensor (shape = (H,W))
                assert(n<N && c<C);
                return &data[n*(C*H*W) + c*(H*W)];
            }

            void displayShape(std::string tName = ""){
                std::cout << "Shape of tensor " << tName <<" : (" <<N << ", "<<C<<", "<<H<<", "<<W<<")\n";
            }

            void show(std::string tName){
                displayShape(tName);
                for(auto n=0; n<N; n++){
                    std::cout << "batch: "<<n<<"\n\t";
                    for(auto ch=0; ch<C; ch++){
                        std::cout << "channel: "<<ch<<"\n\t\t";
                        T*xmat = getMatrix(n, ch);
                        for(auto i=0; i<H; i++){
                            for(auto j=0; j<W; j++){
                                std::cout << xmat[i*W + j] << " ";
                            }
                            std::cout << "\n\t\t";
                        }
                        std::cout << "\n\t";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
    };


   
    template<typename T>
    void convTensor(Tensor<T> &result, Tensor<T> &X, T* weights, int K, int C, int R, int S, char mode='s'){
        // Convolve input tensor X (N,C,H,W) with weights tensor (K, C, R, S)
        // tensor result (N, K, H', W')

        for(int n=0; n<result.N; n++){
            for(int c=0;c<result.C;c++){
                // perform channel-wise convolution between input and output and add to result channel
                T* res = result.getMatrix(n, c);
                for(int xc=0; xc<X.C; xc++){
                    T* mat = X.getMatrix(n, xc);
                    T* wmat = &weights[c*(C*R*S) + xc*(R*S)];
                    conv2D::convolution<T>(res, mat, wmat, X.H, X.W, R, S, mode);
                
                }
            }
        }
    }

}


namespace cnn {

    // NCHW input
    // KCRS filter

    template<typename T>
    class convLayer{
        private:
            // define conv layer data members
            int K, C, R, S;
            // T* weights;
            tensor::Tensor<T> W;
            char mode;
    
        public:
            convLayer(int k, int c, int r, int s, char convMode ='s', char init='r',int fillval=1):
            K(k), C(c), R(r), S(s), W(K, C, R, S, init, fillval), mode(convMode) 
            {}   

            ~convLayer(){
                // destructor - not required as we use W tensor which uses smart pointers for data 
                // delete[] weights;
            }

            T* getWeights(){
                return W.data.get();
            }

            void show(std::string clName){
                std::cout << "Shape of conv Layer " << clName <<" : (" <<K << ", "<<C<<", "<<R<<", "<<S<<")\n";

            }

            void validateInputTensor(int xc){
                // check if input channels match filter channels 
                if(xc != C){
                    std::ostringstream oss;
                    oss << "Input argument C ("<< xc <<") must match C in conv layer ("<< C <<")";
                    throw std::invalid_argument(oss.str());               
               }
            }

            void setFWDargs(tensor::Tensor<T> &result, tensor::Tensor<T> &X){
                // set forward method arguments
                result.N = X.N;
                result.C = K;
                if(mode=='s'){
                    result.H = X.H;
                    result.W = X.W;
                }else if(mode=='v'){
                    result.H = X.H - R + 1;
                    result.W = X.W - S + 1;
                    if(result.H<=0 or result.W<=0){
                        std::ostringstream oss;
                        oss << "Invalid output dimensions: H = " << result.H << ", W = " << result.W
                << ". Input matrix is too small for the layer filter size!";
                        throw std::invalid_argument(oss.str()); 
                    }
                }            
            }

            tensor::Tensor<T> forward(tensor::Tensor<T> &X){
                // check if input tensor channels match filter channels
                tensor::Tensor<T> result;
                try{
                    validateInputTensor(X.C);
                    result = tensor::Tensor<T>(X.N, K, X.H, X.W, 'z');
                    setFWDargs(result, X);
                }catch (const std::invalid_argument& e) {
                    std::cerr << "\nError: " << e.what() << std::endl;
                    std::exit(1);
                }catch (const std::exception& e) {
                    std::cerr << "\nUnexpected error: " << e.what() << std::endl;
                    std::exit(1);
                }
                // Perform convolutions for each input in batch, for each channel
                // C channels -> K channels for each batch element
                tensor::convTensor(result, X, W.data.get(), K, C, R, S, mode);
                return std::move(result);

            }
    };

    template<typename T>
    class ReluLayer{
        
    };

    template<typename T>
    class MLPLayer{

    };

}

#endif