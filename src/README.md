# Design ideology

All functions/classes are defined as templates.
By default it is to be assumed that T corresponds to some typename unless specified otherwise.

## Matrices

Matrices are defined as raw arrays.
There are methods provided to create matrices, view them.

### Functions

- createMaitrx:
    arguments -> (T*) (int x0, int x1, T fillval)
    - x0 : height of matrix
    - x1 : width of matrix
    - fillval : initalization of each data with fillval (default 0, if(-1) -> random values are used (This is a poor design and will be updated!) );

- displayMatrix: arguments -> (void) (T*x, int x0, int x1)
    - x : pointer to matrix 
    - x0 : height
    - x1 : width

## conv2D

This namespace wraps 2D convolution operations. Currently only two types are supported - "same" (shape(y) = shape(x)) & "vaid" (shape(y)[k] = shape(x)[k] - shape(w)[k] + 1 : k={0,1})

### Functions

- convolution:
    arguments -> (void) (T* Y, T *X, T *F, int x0, int x1, int f0, int f1, char mode='s')
    - Y : output matrix
    - X : input matrix
    - F : filter matrix
    - x0 : input height
    - x1 : input width
    - f0 : filter height
    - f1 : filter width
    - mode : 's' for same 'v' for valid convolution ('s' is default)


- convSame:
    follows style of convolution

- convValid:
    follows style of convolution


## Tensors

This namespace wraps tensor objects and tensor convolutions

A tensor uses the following indexing standard: (N, C, H, W)

N: batch size 
C: channel size
H: height 
W: width

### Functions

- createTensor: arguments -> `(std::unique_ptr<T[]>) (int N, int C, int H, int W, chat state = 'c', int fillval = 1)`

- convTensor: arguments -> `(void) (Tensor<T> & result, Tensor<T> &X, T* weights, int K, int C, int R, int S, char mode='s)`
    - Performs tensor convolution between X (input) and weights (filter) and stores it in Y (output)
    - shape of output based on convolution mode
    - result : output tensor
    - X : input tensor
    - weights : pointer to filter data
    - K : number of filters
    - C : channels per filter
    - R : height of filter
    - S : width of filter
    - mode : 's' for same conv, 'v' for valid conv


### Classes

- Tensor: 
    - data members:
    ``` 
     int N, (number of batches)
     int C, (number of channels)
     int H, (height)
     int W, (width)
     std::unique_ptr<T[]> (data); (points to tensor data)
     ```
    
    - Constructor:
    ```cpp

     Tensor(int n, int c, int h, int w, int state='z', int fillval=1) {
                // constructor
                N = n;
                C = c;
                H = h;
                W = w;
                data = createTensor<T>(N, C, H, W, state, fillval);
            }

    ```
    - function members:
        - get -> `(T) (int n=0, int c=0, int h=0, int w=0)` 
            - n : index of batch
            - c : index of channel
            - h : index of height
            - w : index of width
            - returns data at coordinates `(n, c, h, w)`
        
        - getMatrix -> `(T*) (int n, int c)`
            - n : index of batch
            - c : index of channel
            - returns pointer to matrix at coordinates `(n, c)`

        - displayShape -> `(void) (std::string tName = "")`
            - tName : utilty name given to tensor for printing (plan to modify to be data member for reusability)
            - prints shape of tensor `(n, c, h, w)`
        
        - show -> `(void) (std::string tName = ")`
            - tName : utilty name given to tensor for printing (plan to modify to be data member for reusability)
            - displayes matrices grouped by `n`, `c` in the same order
        
    
## CNN
