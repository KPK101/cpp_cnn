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

	cnn::convLayer<tdtype> cl_v = cnn::convLayer<tdtype>(c,c,r,s,'v', 'f', 1);
	cnn::convLayer<tdtype> cl_s = cnn::convLayer<tdtype>(c,c,r,s,'s', 'f', 1);
	
	tensor::Tensor<tdtype> Y = cl_v.forward(X);
	Y = cl_s.forward(Y);
	Y = cl_s.forward(Y);
	Y.displayShape("Y");


}