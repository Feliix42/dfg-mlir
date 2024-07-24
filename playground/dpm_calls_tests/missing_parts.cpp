/*
func.func private @source() -> (i64, i64, i64)
func.func private @sum(i64, i64) -> i64
func.func private @mul(i64, i64) -> i64
func.func private @sink(i64) -> ()
*/

#include <stdint.h>

#include <iostream>

extern "C" {

int64_t sum(int64_t in, int64_t out){
	std::cout << in << "/" << out << std::endl;
	return in+out;
}

int64_t mul(int64_t in, int64_t out){
	return in*out;
}

void sink(int64_t in){
	std::cout << "Sink found item: " << in << std::endl;
	return;
}


struct Triple {
	int64_t a;
	int64_t b;
	int64_t c;
};

void test(const char* string){
	printf("%s", string);
}

void second(){
	test("hello");
}

int64_t dummy(){
	return 3;
}


void takePtr(char* testInt){
	printf("%x", testInt);
}

class Testclass{
public:
	Testclass(int _x, int _y, int _z, float a) : x(_x), y(_y), z(_z), a(a) {}

private:
	int x,y,z;
	float a;
};

Testclass getInt(){
	return Testclass{1,2,3, 0.2};
}

void createPtr() {
	Testclass testInt = getInt();

	char* byteData = (char*)&testInt;

	takePtr(byteData);
}

}
