#include"conv.h"
#include<iostream>

int main(){
	tensor::Tensor<float> X(1, 1, 3, 3, 'r');
	tensor::Tensor<float> W(1, 1, 2, 2);
	std::cout << W.get(0, 0, 1, 1) << std::endl;
    return 0;
}