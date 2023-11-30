#include <stdio.h>
#include <stdint.h>

void algo();

int64_t sum(int64_t a, int64_t b) {
    int64_t res = a + b;

    printf("sum: %ld\n", res);
    return res;
}

int64_t mul(int64_t a, int64_t b) {
    int64_t res = a * b;

    printf("sum: %ld\n", res);
    return res;
}

// int main() {
//     algo();
//     return 0;
// }
