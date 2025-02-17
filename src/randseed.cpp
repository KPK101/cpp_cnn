
#include<random>
#include<iostream>
#include<chrono>

int main(){
    std::random_device rd;
    auto seed = rd() ^ std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    std::cout << "Seed: " << seed << std::endl;

}