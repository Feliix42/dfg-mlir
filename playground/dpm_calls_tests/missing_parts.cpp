/*
func.func private @source() -> (i64, i64, i64)
func.func private @sum(i64, i64) -> i64
func.func private @mul(i64, i64) -> i64
func.func private @sink(i64) -> ()
*/

#include <stdint.h>

#include <iostream>

extern "C" {

int64_t mul(int64_t in, int64_t out){
	return in*out;
}

void sink(int64_t in){
	std::cout << "Sink found item: " << in << std::endl;
	return;
}

}
