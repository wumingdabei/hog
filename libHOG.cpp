#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()
#include <opencv2/opencv.hpp> //only used for file I/O

#include "SimpleImg.hpp"
#include "libHOG.h"
#include "helpers.h"
#include "libHOG_kernels.h"
#include "voc5_reference_kernels.h"

using namespace std;

//@param use_voc5_impl: if true, voc5-based impl. else, use libHOG native impl
libHOG::libHOG(int nLevels_, int interval_, bool use_voc5_impl_){
    use_voc5_impl = use_voc5_impl_;
    nLevels = nLevels_;
    interval = interval_;
    reset_timers();
}

//TODO: take sbin, padding, <ffld or voc5>, <L1 or L2>
libHOG::libHOG(int nLevels_, int interval_){
    use_voc5_impl = false; //use libHOG's native impl
    nLevels = nLevels_;
    interval = interval_;
    reset_timers(); 
}

//empty constructor w/ default settings
libHOG::libHOG(){
    use_voc5_impl = false; //use libHOG's native impl
    nLevels = 40;
    interval = 10;
    reset_timers();
}

libHOG::~libHOG(){
    free_vec_of_ptr(hogBuffer_blocks); //free any leftovers from final call to compute_pyramid()
}

//number of pixels (per channel)
//GB of data output from histogram
double get_hist_gb(vector< int > hogHeight, vector< int > hogWidth){
    int hist_bytes = 0;
    for(int s=0; s<hogHeight.size(); s++){
        hist_bytes += hogHeight[s] * hogWidth[s] * 18 * 4; //18 bins, 4-byte float data
    }
    double hist_gb = ((double)hist_bytes) / 1e9;
    return hist_gb;
}

//hand-coded impl of pyramid. (will modularize it better eventually)
// typical: img pyramid: 16.901926 ms, gradients: 4.586978 ms, hist: 9.008650 ms, norm: 6.529126 ms
//@param img_Mat = image in OpenCV data format
cv::Mat libHOG::compute_pyramid(cv::Mat img_Mat){
   
    //cleanup results from previous call to compute_pyramid()
    //free_vec_of_ptr(hogBuffer_blocks); //it's ok if hogBuffer isn't allocated yet
    
    //hogH.resize(0);
    //hogW.resize(0);    
    libHOG_kernels lHog; //libHOG_kernels constructor initializes lookup tables & constants (mostly for orientation bins)
    
    double hist_gb = 0;

    double start_time = read_timer();

    //TODO: put n_iter OUTSIDE of this function. 
    //for(int iter=0; iter<n_iter; iter++){ //do several runs, take the avg time

//new step
       
		// d image dimension -> gray image d = 1
		// h, w -> height, width of image
		// full -> ??
		// I -> input image, M, O -> mag, orientation OUTPUT
		int h = img_Mat.rows, w = img_Mat.cols, d = 1;
        //int sbin = 4;
        int bin_size = 4;
        int hb = h / bin_size, wb = w / bin_size;
        int ALIGN_IN_BYTES=32;
		
        int stride = compute_stride(w, sizeof(uint8_t), ALIGN_IN_BYTES); 
        
		uint8_t *I =(uint8_t *)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
        memset(I, 0, h * stride * d * sizeof(uint8_t));
		


		for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
			I[y * stride + x ] = img_Mat.at<float>(y, x);
		  }
		}
		
        

        int16_t *M = (int16_t*)malloc_aligned(32, h * stride * d * sizeof(int16_t));
        uint8_t *O = (uint8_t*)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
        memset(M, 0, h * stride * d * sizeof(int16_t));
        memset(O, 0, h * stride * d * sizeof(uint8_t));
        float *hogB = (float *)malloc_aligned(32, hb * wb *32* sizeof(float));
        printf("   ccccccccccccccccccccccc \n");   
        memset(hogB, 0, hb * wb *  sizeof(float));
        	printf("   ddddddddddddddddddddddddd \n");
	



//step 2: gradients
        double grad_start = read_timer();

        //#pragma omp parallel for
        lHog.gradient(h, w, stride, d, d, I, O, M);
        grad_time += (read_timer() - grad_start);
printf("   sssssssssssssssssssssssssa \n");       
//step 3: histogram cells
        double hist_start = read_timer();

        //#pragma omp parallel for
		
		lHog.computeCells_gather(h, w, stride, bin_size,O, M, hb, wb, hogB);
        hist_time += (read_timer() - hist_start);
 printf("   aaaaaaaaaaaaaaaaaaaaaaaaaaaa \n");   
//step 4: normalize cells into blocks
        double norm_start = read_timer();

        //#pragma omp parallel for
        float *hogB_blocks = (float *)malloc_aligned(32, hb * wb * 32 * sizeof(float));
        float *normI = (float *)malloc_aligned(32, hb * wb * sizeof(float));
       	lHog.hogCell_gradientEnergy(hogB, hb, wb, normI);
		lHog.normalizeCells(hogB, normI, hogB_blocks, hb, wb); 
        
        norm_time += (read_timer() - norm_start);
         printf("   rrrrrrrrrrrrrrrrrrrrrrrrrrrr \n");   
//step 5: 
		
		const int hogDepth = 32;
        int gg = 0;
    	cv::Mat res = cv::Mat(hogDepth, (hb-2)*(wb-2), CV_32FC1);
        
          for (int i = 0; i < hogDepth; ++i) {
          // output cols-by-cols
          for (int x = 0; x < hb-1; ++x) {
           for (int y = 0; y < wb-1; ++y) {
                   //printf("eeeeeeeeeeeeeeeeeeeeeeeeee \n");

                 res.at<float>(i, y*wb+x) = hogB_blocks[(y*wb+x)*i];
                gg++;

                         //printf("fffffffffffffffffffffffff \n");   
                         printf("gggggggggggggggggggggg:%d \n",gg);
                         // printf("gggggggggggggggggggggg:%d %d \n",x,y);
                }
          }
         }
    // clean
    //delete[] I;
    //delete[] M;
    //delete[] O;
    //delete[] hogB_blocks;
    free(normI);
    free(hogB);
    free(hogB_blocks);
        
        return res;


}

void libHOG::gen_img_pyra(cv::Mat img_Mat){
    double img_pyra_start = read_timer();

    assert( nLevels == 4*interval ); //TODO: relax this.

	
}


//assumes gen_img_pyra() has been called.


void libHOG::reset_timers(){
    img_pyra_time = 0;
    alloc_time = 0;
    grad_time = 0;
    hist_time = 0;
    norm_time = 0;
}

//HACK: assumes we're using the protocol "sbin=4 for top octave; sbin=8 otherwise"

