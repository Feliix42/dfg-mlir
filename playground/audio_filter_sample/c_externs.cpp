
#include <kissfft/kiss_fftr.h>
//#include <sndfile.h>
#include <array>
#include <tuple>
#include <random>
#include <iostream>
#include <cmath>

const int SAMPLE_RATE = 44100;
const int TOTAL_SAMPLES = 65535;

const int SEGMENT_SIZE = 1024;

const int FFT_SIZE = SEGMENT_SIZE / 2 + 1;

const int HOP_SIZE = SEGMENT_SIZE / 2; // a lot of overlap
const float CUTOFF_FREQUENCY = 2000.0f;

/*
void exportAsWAV(const std::array<float, TOTAL_SAMPLES> &samples, const std::string& fileName) {
	SF_INFO sfinfo;
	sfinfo.channels = 1;
	sfinfo.samplerate = SAMPLE_RATE;
	sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

	SNDFILE* file = sf_open(fileName.c_str(), SFM_WRITE, &sfinfo);
	if(file) {
		sf_write_float(file, samples.data(), samples.size());
		sf_close(file);
	}
}
*/
void applyFFT(const std::array<float, SEGMENT_SIZE>& input, std::array<kiss_fft_cpx, FFT_SIZE> &output) {
	kiss_fftr_cfg cfg = kiss_fftr_alloc(SEGMENT_SIZE, 0, nullptr, nullptr);
	if(!cfg) {
		throw std::runtime_error("Fail");
	}
	kiss_fftr(cfg, input.data(), output.data());
	kiss_fftr_free(cfg);
}

void applyLowPassFilter(std::array<kiss_fft_cpx, FFT_SIZE> &freq_data, float cutoff_frequency) {
	int cutoff_index = static_cast<int>(cutoff_frequency / SAMPLE_RATE * SEGMENT_SIZE);

	for(int i = cutoff_index; i < FFT_SIZE; i++){
		freq_data[i].r = 0;
		freq_data[i].i = 0;
	}
}

void applyIFFT(std::array<kiss_fft_cpx, FFT_SIZE> &freq_data, std::array<float, SEGMENT_SIZE> &output) {
	kiss_fftr_cfg cfg = kiss_fftr_alloc(SEGMENT_SIZE, 1, nullptr, nullptr);
	if(!cfg) {
		throw std::runtime_error("Fail");
	}
	kiss_fftri(cfg, freq_data.data(), output.data());
	// normalize
	for(auto &sample: output) {
		sample /= SEGMENT_SIZE;
	}
	kiss_fftr_free(cfg);
}

void applyHammingWindow(std::array<float, SEGMENT_SIZE> &window) {
	for(size_t i = 0; i < window.size() ; i++){
		window[i] *= 0.54 - 0.46 * cos(2 * M_PI * i / (window.size() - 1));
	}
}

bool data_is_init = false;
std::array<float, TOTAL_SAMPLES> global_data;

extern "C" void fill_data(float *data){
	if(!data_is_init) {
		data_is_init = true;
		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
		for(size_t i = 0 ; i < TOTAL_SAMPLES ; i++) {
			global_data[i] = distribution(generator);
		}
		std::cout << "Gen done\n";
	}
	for(size_t i = 0 ; i < TOTAL_SAMPLES ; i++) {
		data[i] = global_data[i];
	}
}


extern "C" void filter_data(std::array<float, TOTAL_SAMPLES> *input, std::array<float, TOTAL_SAMPLES> *output) {
	std::array<float, SEGMENT_SIZE> windowed_segment;
	std::array<kiss_fft_cpx, FFT_SIZE> freq_data;
	std::array<float, SEGMENT_SIZE> ifft_segment;
	for(size_t i = 0 ; i + SEGMENT_SIZE <= input->size() ; i+= HOP_SIZE) {
		std::copy(input->begin() + i, input->begin() + i + SEGMENT_SIZE, windowed_segment.begin());
		applyHammingWindow(windowed_segment);
		applyFFT(windowed_segment, freq_data);
		applyLowPassFilter(freq_data, CUTOFF_FREQUENCY);
		applyIFFT(freq_data, ifft_segment);
		// write to output
		for(size_t j = 0 ; j < SEGMENT_SIZE ; j++){
			(*output)[i+j] += ifft_segment[j];
		}

	}


}

extern "C" void push_done() {
	std::cout << "Push done\n";
}
