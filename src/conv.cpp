#include"conv.h"
#include<iostream>

int main(){
	using tdtype = float;
	size_t h = 3;
	size_t w = 3;
	size_t c = 1;
	size_t r = 1;
	size_t s = 1;

	tensor::Tensor<tdtype> X(1, c, h, w, 'r');
	tensor::Tensor<tdtype> W(1, c, r, s);


	cnn::convLayer cl = cnn::convLayer<tdtype>(1,c,r,s,'s');

	std::cout << "Displaying weights of conv layer :\n\n";
	tdtype* weights = cl.getWeights();
	for(auto i=0; i<r; i++){
		for(auto j=0; j<s; j++){
			std::cout << weights[i*s + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	/////////////////////////////////////////////////////

	std::cout << "Displaying input of conv:\n\n";
	tdtype *xmat = X.getMatrix(0, 0);
	for(auto i=0; i<h; i++){
		for(auto j=0; j<w; j++){
			std::cout << xmat[i*w + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	/////////////////////////////////////////////////////
	tensor::Tensor<tdtype> Y = cl.forward(X);

	std::cout << "Displaying output of conv:\n\n";
	tdtype *ymat = Y.getMatrix(0, 0);
	for(auto i=0; i<h; i++){
		for(auto j=0; j<w; j++){
			std::cout << ymat[i*w + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
    return 0;
}