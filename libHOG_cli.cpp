#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <piotr_fhog.hpp>
#include "SimpleImg.hpp"
#include <libHOG.h>

using namespace std;

int main (int argc, char **argv)
{
    //TODO: take image path as input
	
    cv::Mat img = cv::imread("images_640x480/carsgraz_001.image.jpg");
    
	//FHoG::extract();
	
    libHOG libtest;
    //cv::Mat res = cv::Mat(32, 18644, CV_32FC1);
    cv::Mat img_out = libtest.compute_pyramid(img); 
     
    //TODO: save output HOG as CSV
    //cv::imshow("test", img_out);
    
      FILE *fp ;
        fp = fopen( "myfile.txt", "w" ) ;

        for(int i=0; i<img_out.rows-1; i++){
            printf("rrrrrrrrrrrrrrrrrrrrrrr:%d \n",i);
           for(int j=0; j<img_out.cols; j++){
                fprintf(fp, "%3.0f", img_out.at<float>(i,j) ) ;  // the data type should be matched.

                                                                                          //  the same as that of my_mat
           }
           fprintf( fp, "\n" ) ;
        }
        fclose( fp ) ;

    return 0;
}
