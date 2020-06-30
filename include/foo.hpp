#pragma once

class foo {
public:
    class foo_1{
        public:
            int var1 = 1;
            void hello1();    

    };

    class foo_2{
        public:
            int var2 =2;
            void hello2();    

    };

static void hello();
};



class geom{
    protected:
        double r=0;
        double a=0;
        double b=0;
        double h=0;

    private:
        void triangle (double, double);
    public:
        void circle( double );
        void rectangle( double, double );
        
};


class area : public geom {
    public:

    double rectarea( double, double);    

};


