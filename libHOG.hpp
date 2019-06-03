/*
   - c++ wrapper for the piotr toolbox
   Created by Tomas Vojir, 2014
 */

#ifndef FHOG_HEADER_7813784354687
#define FHOG_HEADER_7813784354687

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "gradientMex.h"
#include "libHOG_kernels.h"

static libHOG_kernels lHog;

class FHoG {
 public:
  // description: extract hist. of gradients(use_hog == 0), hog(use_hog == 1) or
  // fhog(use_hog == 2)
  // input: float one channel image as input, hog type
  // return: computed descriptor
  static cv::Mat extract(const cv::Mat &img, int use_hog = 2, int bin_size = 4,
                         int n_orients = 9, int soft_bin = -1,
                         float clip = 0.2) {
    // d image dimension -> gray image d = 1
    // h, w -> height, width of image
    // full -> ??
    // I -> input image, M, O -> mag, orientation OUTPUT
    int h = img.rows, w = img.cols, d = 1;
    bool full = true;
    if (h < 2 || w < 2) {
      std::cerr << "I must be at least 2x2." << std::endl;
      return cv::Mat();
    }

    // image cols-by-cols
    float *I = new float[h * w];
    for (int x = 0; x < w; ++x) {
      for (int y = 0; y < h; ++y) {
        I[x * h + y] = img.at<float>(y, x);
      }
    }

    float *M = new float[h * w], *O = new float[h * w];
    // memset(M, 0, h*w*sizeof(float));
    // memset(O, 0, h*w*sizeof(float));
    gradMag(I, M, O, h, w, d, full);
    delete[] I;

    int n_chns = (use_hog == 0)
                     ? n_orients
                     : (use_hog == 1 ? n_orients * 4 : n_orients * 3 + 5);
    int hb = h / bin_size, wb = w / bin_size;

    float *H = new float[hb * wb * n_chns];
    memset(H, 0, hb * wb * n_chns * sizeof(float));

    if (use_hog == 0) {
      full = false;  // by default
      gradHist(M, O, H, h, w, bin_size, n_orients, soft_bin, full);
    } else if (use_hog == 1) {
      full = false;  // by default
      hog(M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip);
    } else {
      fhog(M, O, H, h, w, bin_size, n_orients, soft_bin, clip);
    }

    // convert, assuming row-by-row-by-channel storage
    int n_res_channels = (use_hog == 2)
                             ? n_chns - 1
                             : n_chns;  // last channel all zeros for fhog
    cv::Mat res = cv::Mat(cv::Size(hb * wb, n_res_channels), CV_32F);

    for (int i = 0; i < n_res_channels; ++i) {
      // output cols-by-cols
      for (int x = 0; x < wb; ++x) {
        for (int y = 0; y < hb; ++y) {
          res.at<float>(i, y * wb + x) = H[i * hb * wb + x * hb + y];
        }
      }
    }

    // clean
    delete[] M;
    delete[] O;
    delete[] H;

    return res;
  }

  static cv::Mat inline extract_raw(const cv::Mat &img, int use_hog = 2,
                                    int bin_size = 4, int n_orients = 9,
                                    int soft_bin = -1, float clip = 0.2) {
    // d image dimension -> gray image d = 1
    // h, w -> height, width of image
    // full -> ??
    // I -> input image, M, O -> mag, orientation OUTPUT
    int h = img.rows, w = img.cols, d = 1;
    bool full = true;
    if (h < 2 || w < 2) {
      std::cerr << "I must be at least 2x2." << std::endl;
      return cv::Mat();
    }

    // image cols-by-cols
    float *I = new float[h * w];
    for (int x = 0; x < w; ++x) {
      for (int y = 0; y < h; ++y) {
        I[x * h + y] = img.at<float>(y, x);
      }
    }

    float *M = new float[h * w], *O = new float[h * w];
    // memset(M, 0, h*w*sizeof(float));
    // memset(O, 0, h*w*sizeof(float));

    gradMag(I, M, O, h, w, d, full);
    delete[] I;

    int n_chns = (use_hog == 0)
                     ? n_orients
                     : (use_hog == 1 ? n_orients * 4 : n_orients * 3 + 5);
    int hb = h / bin_size, wb = w / bin_size;

    float *H = new float[hb * wb * n_chns];
    memset(H, 0, hb * wb * n_chns * sizeof(float));

    if (use_hog == 0) {
      full = false;  // by default
      gradHist(M, O, H, h, w, bin_size, n_orients, soft_bin, full);
    } else if (use_hog == 1) {
      full = false;  // by default
      hog(M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip);
    } else {
      fhog(M, O, H, h, w, bin_size, n_orients, soft_bin, clip);
    }

    // convert, assuming row-by-row-by-channel storage

    int n_res_channels = (use_hog == 2)
                             ? n_chns - 1
                             : n_chns;  // last channel all zeros for fhog

    cv::Mat feat = cv::Mat(hb, wb, CV_32FC(n_res_channels), H);

    cv::Mat res = feat.clone();
    // clean
    delete[] M;
    delete[] O;
    delete[] H;

    return res;
  }

  static const int ALIGN_IN_BYTES = 32;
  static cv::Mat inline fast_extract(const cv::Mat &img, int use_hog = 2,
                                     int bin_size = 4, int n_orients = 9,
                                     int soft_bin = -1, float clip = 0.2) {
    // d image dimension -> gray image d = 1
    // h, w -> height, width of image
    // full -> ??
    // I -> input image, M, O -> mag, orientation OUTPUT
    int h = img.rows, w = img.cols, d = 1, stride;
    if (h < 2 || w < 2) {
      std::cerr << "I must be at least 2x2." << std::endl;
      return cv::Mat();
    }

    stride = lHog.compute_stride(img.cols, sizeof(uint8_t), ALIGN_IN_BYTES);

    uint8_t *I =
        (uint8_t *)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
    memset(I, 0, h * stride * d * sizeof(uint8_t));

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int channel = 0; channel < d; channel++) {
          // this->data[y*stride*3 + x*3 + channel] =
          // img.at<cv::Vec3b>(y,x)[channel];  //channels as inner dimension
          I[y * stride + x + channel * stride * h] =
              img.at<uchar>(y, x);  // channels as outer dimension
          // this->data[y*stride + x + channel*stride*height] =
          // img.data[y*width*3 + x*width + channel];
        }
      }
    }

    int16_t *M =
        (int16_t *)malloc_aligned(32, h * stride * d * sizeof(int16_t));
    memset(M, 0, h * stride * d * sizeof(int16_t));
    uint8_t *O =
        (uint8_t *)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
    memset(O, 0, h * stride * d * sizeof(uint8_t));

    int n_chns = (use_hog == 0)
                     ? n_orients
                     : (use_hog == 1 ? n_orients * 4 : n_orients * 3 + 5);
    int hb = h / bin_size, wb = w / bin_size;

    float *H = (float *)malloc_aligned(32, hb * wb * n_chns * sizeof(float));
    memset(H, 0, hb * wb * n_chns * sizeof(float));

    lHog.gradient(h, w, stride, d, 1, I, O, M);                         // x1
    lHog.computeCells_gather(h, w, stride, bin_size, O, M, hb, wb, H);  // x2

    free(I);
    free(M);
    free(O);

    float *nH = (float *)malloc_aligned(32, hb * wb * n_chns * sizeof(float));
    memset(nH, 0, hb * wb * n_chns * sizeof(float));

    float *nI = (float *)malloc_aligned(32, hb * wb * sizeof(float));

    lHog.hogCell_gradientEnergy(H, hb, wb, nI);  // x0.5
    lHog.normalizeCells(H, nI, nH, hb, wb);      // 0.75

    // convert, assuming row-by-row-by-channel storage
    int n_res_channels = (use_hog == 2)
                             ? n_chns - 1
                             : n_chns;  // last channel all zeros for fhog

    cv::Mat res = cv::Mat(cv::Size(hb * wb, n_res_channels), CV_32F);

    for (int i = 0; i < n_res_channels; ++i) {
      // output cols-by-cols
      for (int x = 0; x < wb; ++x) {
        for (int y = 0; y < hb; ++y) {
          res.at<float>(i, y * wb + x) =
              nH[x * (n_res_channels + 1) + y * wb * (n_res_channels + 1) + i];
        }
      }
    }

    // clean
    free(nI);
    free(H);
    free(nH);

    return res;
  }

  static cv::Mat inline fast_extract_raw(const cv::Mat &img, int use_hog = 2,
                                         int bin_size = 4, int n_orients = 9,
                                         int soft_bin = -1, float clip = 0.2) {
    // d image dimension -> gray image d = 1
    // h, w -> height, width of image
    // full -> ??
    // I -> input image, M, O -> mag, orientation OUTPUT
    int h = img.rows, w = img.cols, d = 1, stride;
    if (h < 2 || w < 2) {
      std::cerr << "I must be at least 2x2." << std::endl;
      return cv::Mat();
    }

    stride = lHog.compute_stride(img.cols, sizeof(uint8_t), ALIGN_IN_BYTES);

    uint8_t *I =
        (uint8_t *)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
    memset(I, 0, h * stride * d * sizeof(uint8_t));

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int channel = 0; channel < d; channel++) {
          // this->data[y*stride*3 + x*3 + channel] =
          // img.at<cv::Vec3b>(y,x)[channel];  //channels as inner dimension
          I[y * stride + x + channel * stride * h] =
              img.at<uchar>(y, x);  // channels as outer dimension
          // this->data[y*stride + x + channel*stride*height] =
          // img.data[y*width*3 + x*width + channel];
        }
      }
    }

    int16_t *M =
        (int16_t *)malloc_aligned(32, h * stride * d * sizeof(int16_t));
    memset(M, 0, h * stride * d * sizeof(int16_t));
    uint8_t *O =
        (uint8_t *)malloc_aligned(32, h * stride * d * sizeof(uint8_t));
    memset(O, 0, h * stride * d * sizeof(uint8_t));

    int n_chns = (use_hog == 0)
                     ? n_orients
                     : (use_hog == 1 ? n_orients * 4 : n_orients * 3 + 5);
    int hb = h / bin_size, wb = w / bin_size;

    float *H = (float *)malloc_aligned(32, hb * wb * n_chns * sizeof(float));
    memset(H, 0, hb * wb * n_chns * sizeof(float));

    lHog.gradient(h, w, stride, d, 1, I, O, M);                         // x1
    lHog.computeCells_gather(h, w, stride, bin_size, O, M, hb, wb, H);  // x2

    free(I);
    free(M);
    free(O);

    float *nH = (float *)malloc_aligned(32, hb * wb * n_chns * sizeof(float));
    memset(nH, 0, hb * wb * n_chns * sizeof(float));

    float *nI = (float *)malloc_aligned(32, hb * wb * sizeof(float));

    lHog.hogCell_gradientEnergy(H, hb, wb, nI);  // x0.5
    lHog.normalizeCells(H, nI, nH, hb, wb);      // 0.75

    // convert, assuming row-by-row-by-channel storage
    int n_res_channels = (use_hog == 2)
                             ? n_chns - 1
                             : n_chns;  // last channel all zeros for fhog

    cv::Mat res = cv::Mat(hb, wb, CV_32FC(n_res_channels));

    float *pRes = (float *)(res.data);

    for (int i = 0; i < n_res_channels; ++i) {
      size_t channel_idx = wb * hb * i;
      // output cols-by-cols
      for (int x = 0; x < wb; ++x) {
        size_t w_idx = x * (n_res_channels + 1);
        for (int y = 0; y < hb; ++y) {
          pRes[channel_idx + y * wb + x] =
              nH[w_idx + y * wb * (n_res_channels + 1) + i];
        }
      }
    }

    // clean
    free(nI);
    free(H);
    free(nH);

    return res;
  }
};

#endif  // FHOG_HEADER_7813784354687
