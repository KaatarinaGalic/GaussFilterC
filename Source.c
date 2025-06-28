#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gaussian kernel
double* create_gaussian_kernel(int radius) {
    int size = 2 * radius + 1;
    double sigma = radius / 3.0;
    if (sigma == 0.0) sigma = 1.0;

    double* kernel = (double*)malloc(size * sizeof(double));
    if (!kernel) {
        printf("Greška: Nema dovoljno memorije za kernel.\n");
        exit(1);
    }

    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double value = exp(-(i * i) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
        kernel[i + radius] = value;
        sum += value;
    }

    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Sekvencijalni blur
void horizontal_blur_seq(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, double* kernel) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            for (int c = 0; c < channels; c++) {
                double sum = 0.0;
                for (int k = -radius; k <= radius; k++) {
                    int px = x + k;
                    if (px < 0) px = 0;
                    if (px >= width) px = width - 1;
                    sum += input[(y * width + px) * channels + c] * kernel[k + radius];
                }
                int val = (int)(sum + 0.5);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[(y * width + x) * channels + c] = (unsigned char)val;
            }
        }
    }
}

void vertical_blur_seq(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, double* kernel) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            for (int c = 0; c < channels; c++) {
                double sum = 0.0;
                for (int k = -radius; k <= radius; k++) {
                    int py = y + k;
                    if (py < 0) py = 0;
                    if (py >= height) py = height - 1;
                    sum += input[(py * width + x) * channels + c] * kernel[k + radius];
                }
                int val = (int)(sum + 0.5);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[(y * width + x) * channels + c] = (unsigned char)val;
            }
        }
    }
}

// Paralelni blur
void horizontal_blur(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, double* kernel) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                double sum = 0.0;
                for (int k = -radius; k <= radius; k++) {
                    int px = x + k;
                    if (px < 0) px = 0;
                    if (px >= width) px = width - 1;
                    sum += input[(y * width + px) * channels + c] * kernel[k + radius];
                }
                int val = (int)(sum + 0.5);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[(y * width + x) * channels + c] = (unsigned char)val;
            }
        }
    }
}

void vertical_blur(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, double* kernel) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                double sum = 0.0;
                for (int k = -radius; k <= radius; k++) {
                    int py = y + k;
                    if (py < 0) py = 0;
                    if (py >= height) py = height - 1;
                    sum += input[(py * width + x) * channels + c] * kernel[k + radius];
                }
                int val = (int)(sum + 0.5);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[(y * width + x) * channels + c] = (unsigned char)val;
            }
        }
    }
}

int main() {
    const char* input_path = "Slike/converted-image-_1_.bmp";
    const char* output_seq_path = "Slike/zamagljena_sekvencijalno.bmp";
    const char* output_par_path = "Slike/zamagljena_paralelno.bmp";

    int width, height, channels;
    unsigned char* input_image = stbi_load(input_path, &width, &height, &channels, 3);
    if (!input_image) {
        printf("Greška: Ne mogu uèitati sliku.\n");
        return 1;
    }

    printf("Uèitana slika: %d x %d\n", width, height);
    printf("Kanali: %d\n", 3);
    printf("OpenMP niti (maks): %d\n", omp_get_max_threads());

    int radius = 50;
    double* kernel = create_gaussian_kernel(radius);

    // PARALELNA OBRADA
    omp_set_num_threads(4);
    unsigned char* temp_par = (unsigned char*)malloc(width * height * 3);
    unsigned char* output_par = (unsigned char*)malloc(width * height * 3);
    double start_par = omp_get_wtime();
    horizontal_blur(input_image, temp_par, width, height, 3, radius, kernel);
    vertical_blur(temp_par, output_par, width, height, 3, radius, kernel);
    double end_par = omp_get_wtime();
    int duration_par = (int)((end_par - start_par) * 1000.0);
    printf("Vrijeme obrade slike paralelno: %dms\n", duration_par);
    stbi_write_bmp(output_par_path, width, height, 3, output_par);

    // SEKVENCIJALNA OBRADA
    unsigned char* temp_seq = (unsigned char*)malloc(width * height * 3);
    unsigned char* output_seq = (unsigned char*)malloc(width * height * 3);
    double start_seq = omp_get_wtime();
    horizontal_blur_seq(input_image, temp_seq, width, height, 3, radius, kernel);
    vertical_blur_seq(temp_seq, output_seq, width, height, 3, radius, kernel);
    double end_seq = omp_get_wtime();
    int duration_seq = (int)((end_seq - start_seq) * 1000.0);
    printf("Vrijeme obrade slike sekvencijalno: %dms\n", duration_seq);
    stbi_write_bmp(output_seq_path, width, height, 3, output_seq);

    // Oslobaðanje memorije
    stbi_image_free(input_image);
    free(kernel);
    free(temp_seq);
    free(output_seq);
    free(temp_par);
    free(output_par);

    return 0;
}
