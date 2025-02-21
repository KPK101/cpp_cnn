#include"conv.h"
#include<iostream>

int main(){

	using tdtype = float;
	size_t h = 3;
	size_t w = 3;
	size_t c = 3;
	size_t r = 2;
	size_t s = 2;

	
	tensor::Tensor<tdtype> X(2, c, h, w, 'r');
	tensor::Tensor<tdtype> W(2, c, r, s, 'r');
	X.displayShape("X");
	W.displayShape("W");
	// std::cout << "W init data @: " << W.data.get() << "\n";
	// tensor::Tensor<tdtype> W1 = std::move(W);
	// tensor::Tensor<tdtype> W2 = W;
	// tensor::Tensor<tdtype> W3(W);

	// std::cout << "W1 init data @: "<< W1.data.get() << "\n";
	// std::cout << "W  data @: " << W.data.get() << "\n";
	// std::cout << "W2 init data @: " << W2.data.get() << "\n";
	// std::cout << "W3 init data @: " << W3.data.get() << "\n";

	cnn::convLayer<tdtype> cl = cnn::convLayer<tdtype>(c,c,r,s,'v', 'f', 1);
	tensor::Tensor<tdtype> Y = cl.forward(X);
	Y = cl.forward(Y);
	Y = cl.forward(Y);
	Y.displayShape("Y");


}