gcc main.c -o als -lm -fopenmp -lpthread -O3
./als als_recsys $1 $2
