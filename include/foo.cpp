#include "foo.hpp"
#include <iostream>

void foo::hello(){
    std::cout << "Hello" << std::endl;
}

void foo::foo_1::hello1(){
    std::cout<< "im in hello "<< this->var1 << std::endl;
}

void foo::foo_2::hello2(){
    std::cout<< "im in hello "<< this->var2 << std::endl;;
}


double area::rectarea(double a1, double a2){

    // this->circle(a1);
    
    // this->b = a2;

    

    


};
