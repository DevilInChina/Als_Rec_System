gcc main.c -o als -lm -fopenmp -lpthread -Ofast
./als als_recsys $1 $2
