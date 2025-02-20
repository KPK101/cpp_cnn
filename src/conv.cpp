#include"conv.h"
#include<iostream>

int main(){
	using tdtype = float;
	size_t h = 3;
	size_t w = 3;
	size_t c = 3;
	size_t r = 1;
	size_t s = 1;

	tensor::Tensor<tdtype> X(2, c, h, w, 'r');

	cnn::convLayer cl = cnn::convLayer<tdtype>(2,c,r,s,'s', 'f', 1);

	std::cout << "Displaying weights of conv layer (all channels):\n\n";
	tdtype* weights = cl.getWeights();
	for(auto i=0; i<r; i++){
		for(auto j=0; j<s; j++){
			std::cout << weights[i*s + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	/////////////////////////////////////////////////////
	X.displayShape("X");
	std::cout << "Displaying input of conv:\n\n";
	for(auto ch=0; ch<c; ch++){
		std::cout << "channel: "<<ch<<"\n\t";
		tdtype *xmat = X.getMatrix(0, ch);
		for(auto i=0; i<h; i++){
			for(auto j=0; j<w; j++){
				std::cout << xmat[i*w + j] << " ";
			}
			std::cout << "\n\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	/////////////////////////////////////////////////////
	tensor::Tensor<tdtype> Y = cl.forward(X);

	Y.displayShape("Y");
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