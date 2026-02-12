#include <stdio.h>
#include <omp.h>
int main() {

int i, sum = 0;
#pragma omp parallel for
for (i = 1; i <= 100; i++) {
sum += i;
}
printf("Sum = %d\n", sum);
return 0;
}