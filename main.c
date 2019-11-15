#include <pthread.h>
#include <fcntl.h>
#include <zconf.h>
#include "mmio_highlevel.h"


void printmat(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%4.2f ", A[i * n + j]);
        printf("\n");
    }
}

void printvec(float *x, int n)
{
    for (int i = 0; i < n; i++)
        printf("%4.2f\n", x[i]);
}


// A is m x n, AT is n x m

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

double getTolTime(struct timeval *t1,struct timeval *t2){
    return (t2->tv_sec - t1->tv_sec) * 1000.0 + (t2->tv_usec - t1->tv_usec) / 1000.0;
}

#ifdef UPDATE_MTX_MULTITHREAD
void updateMtx_part(Para*parameter){
    struct timeval t1, t2;
    float *smat = (float *)malloc(sizeof(float) * parameter->f * parameter->f);
    float *svec = (float *)malloc(sizeof(float) * parameter->f);
    int maxnzr = 0;

    for (int i = parameter->begL; i < parameter->endL; i++){
        maxnzr = max(maxnzr,parameter->Mtx->ia[i+1]-parameter->Mtx->ia[i]); /// n
    }
    float *ri = (float *)malloc(sizeof(float) * maxnzr);
    float *sX = (float *)malloc(sizeof(float) * maxnzr * parameter->f);
    float *sXT = (float *)malloc(sizeof(float) * maxnzr * parameter->f);

    for (int i = parameter->begL; i < parameter->endL; i++)
    {
        gettimeofday(&t1, NULL);
        //printf("\n i = %i", i);
        float *partOfUpdate = parameter->Update+i * parameter->f;

        int begN = parameter->Mtx->ia[i];
        int endN = parameter->Mtx->ia[i+1];
        int nzcur = endN - begN;

        memcpy(ri,parameter->Mtx->val+begN, sizeof(float)*nzcur);/// n*maxNZR

        int count = 0;
        for(int k = begN ; k < endN ; ++k){///maxNZR * f * n
            memcpy(sX+(k-begN) * parameter->f, &parameter->Unchange[parameter->Mtx->ja[k] * parameter->f], sizeof(float) * parameter->f);
        }

        transpose(sXT, sX, nzcur, parameter->f);///n * nzcur * f

        matmat_BtB(smat, sXT, parameter->f, nzcur, parameter->f);//// n*f*nzcur*f*0.5

        for (int j = 0; j < parameter->f; ++j)
            smat[j * parameter->f + j] += parameter->lamda;

        gettimeofday(&t2, NULL);
        parameter->time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        matvec(sXT, ri, svec, parameter->f, nzcur);
        gettimeofday(&t2, NULL);
        parameter->time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        int cgiter = 0;

        cg(smat, partOfUpdate, svec, parameter->f, &cgiter, 100, 0.00001);

        gettimeofday(&t2, NULL);
        parameter->time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }


    free(ri);
    free(sX);
    free(sXT);
    free(smat);
    free(svec);
}

void updateMtx_recsys( sparseMtx *Mtx, float *Unchange, float *Update,
                    int f, float lamda, int begL, int endL,
                      double *time_prepareA, double *time_prepareb, double *time_solver)
{
    int k =  (endL-begL)%THREAD_NUMBERS;
    int totThreads = THREAD_NUMBERS+(k!=0);
    Para *para=malloc(sizeof(Para)*(totThreads));
    pthread_t *pids = malloc(sizeof(pthread_t)*totThreads);
    int block = (endL-begL)/THREAD_NUMBERS;
    for(int i = 0 ; i < totThreads ; ++i){
        para[i].Mtx = Mtx;
        para[i].Unchange = Unchange;
        para[i].Update = Update;
        para[i].f=f;
        para[i].lamda=lamda;
        para[i].begL = begL+i*block;
        para[i].endL = i!=THREAD_NUMBERS?(begL + (i + 1) * block):endL;
        para[i].time_solver=0;
        para[i].time_prepareb=0;
        para[i].time_prepareA=0;
    }
    for(int i = 0 ; i < totThreads ; ++i){
        pthread_create(pids+i,NULL,(void*)updateMtx_part,para+i);
    }
    for(int i = 0 ; i < totThreads; ++i){
        pthread_join(pids[i],NULL);
    }
    for(int i = 0 ; i < totThreads; ++i){
        *time_prepareA+=para[i].time_prepareA/THREAD_NUMBERS;
        *time_prepareb+=para[i].time_prepareb/THREAD_NUMBERS;
        *time_solver+=para[i].time_solver/THREAD_NUMBERS;
    }
    free(para);
    free(pids);
}
#else


void updateMtx_recsys( sparseMtx *Mtx, float *Unchange, float *Update,
                       int f, float lamda, int begL, int endL,
                       double *time_prepareA, double *time_prepareb, double *time_solver)__attribute__((optimize("Ofast")));

void updateMtx_recsys( sparseMtx *Mtx, float *Unchange, float *Update,
                       int f, float lamda, int begL, int endL,
                       double *time_prepareA, double *time_prepareb, double *time_solver){
#pragma omp parallel for
    for (int i = begL; i < endL; i++)
    {

        float *partOfUpdate = Update+i * f;

        int begN = Mtx->ia[i];
        int endN = Mtx->ia[i+1];
        int nzcur = endN - begN;

        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        float *ri = (float *)malloc(sizeof(float) * nzcur);
        float *sX = (float *)malloc(sizeof(float) * nzcur * f);
        float *sXT = (float *)malloc(sizeof(float) * nzcur * f);
        memcpy(ri,Mtx->val+begN, sizeof(float)*nzcur);/// n*maxNZR

        for(int k = begN ; k < endN ; ++k){///maxNZR * f * n
            memcpy(sX+(k-begN) * f, &Unchange[Mtx->ja[k] * f], sizeof(float) * f);
        }

        transpose(sXT, sX, nzcur, f);///n * nzcur * f

        matmat_BtB(smat, sXT, f, nzcur, f);//// n*f*nzcur*f*0.5

        for (int j = 0; j < f; ++j)
            smat[j * f + j] += lamda;

        matvec(sXT, ri, svec, f, nzcur);

        int cgiter = 0;
        cg(smat, partOfUpdate, svec, f, &cgiter, 100, 0.00001);

        free(ri);
        free(sX);
        free(sXT);
        free(smat);
        free(svec);
    }

}
#endif

void als_recsys( sparseMtx*MtxR, float *X, float *Y,
                 int m, int n, int f, float lamda, int nnzR)
{/// @Todo convert R to csr csc
    //

    sparseMtx MtxC;

    csrToCsc(MtxR,&MtxC);
    // create YT and Rp

    float *Rp = (float *)malloc(sizeof(float) * nnzR);

    int iter = 0;
    double error = 0.0;
    double error_old = 0.0;
    double error_new = 0.0;
    struct timeval t1, t2;

    double time_updatex_prepareA = 0;
    double time_updatex_prepareb = 0;
    double time_updatex_solver = 0;

    double time_updatey_prepareA = 0;
    double time_updatey_prepareb = 0;
    double time_updatey_solver = 0;

    double time_updatex = 0;
    double time_updatey = 0;
    double time_validate = 0;
    int tot = 0;
    int  Ylen = n*f;
    double time_one = 0;

    do{/// R sparse X,Y,Rp dense
        // step 1. update X
        time_one = 0;
        gettimeofday(&t1, NULL);

        updateMtx_recsys(MtxR, Y, X, f, lamda, 0, m,
                         &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_one+=(t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 2. update Y
        gettimeofday(&t1, NULL);

        updateMtx_recsys(&MtxC, X, Y, f, lamda, 0, n,
                         &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);


        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_one+=(t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        // step 3. validate
        // step 3-1. matrix multiplication
        gettimeofday(&t1, NULL);

        specMatmat_transB(MtxR,Rp, X, Y, m, f, n);

        // step 3-2. calculate error
        error_new = 0.0;
        int nnz = nnzR;
        for(int i = 0 ; i < nnzR ; ++i){
            error_new+=Rp[i];
        }

        error_new = sqrtf(error_new/nnz);

        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        time_one+=(t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        error = fabs(error_new - error_old) / error_new;

        error_old = error_new;
        printf("iter = %i, error = %f err_new = %f time = %f\n", iter, error,error_new,time_one);

        iter++;
    }
    while(iter < 1000 && error > 0.0001);

    //printf("\nR = \n");
    //printmat(R, m, n);

    //printf("\nRp = \n");
    //printmat(Rp, m, n);

    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);
    printf("Total    %4.2f ms\n",time_updatex+time_updatey+time_validate);
    printf("InCg     %4.2f ms\n",cgbegT);
    free(Rp);

    //mtxDestory(MtxR);
    mtxDestory(&MtxC);
}



int main(int argc, char ** argv)
{
    int f = 0;
    int m, n, nnzR, isSymmetricR;

    // method
    char *method = argv[1];
    printf("\n");

    char *filename = argv[2];
    printf ("filename = %s\n", filename);

    struct timeval t1,t2;

    mmio_info(&m, &n, &nnzR, &isSymmetricR, filename);

    int *csrRowPtrR = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxR = (int *)malloc(nnzR * sizeof(int));
    float *csrValR    = (float *)malloc(nnzR * sizeof(float));
    struct timeval tb,te;
    mmap_io_data(csrRowPtrR, csrColIdxR, csrValR, filename);

    sparseMtx R;
    R.OtherI = malloc(sizeof(int)*nnzR);

    R.n = n;
    R.m = m;
    R.ia = csrRowPtrR;
    R.ja = csrColIdxR;
    R.val = csrValR;
    for(int i = 0 ; i < m ; ++i){
        for(int j = R.ia[i] ; j < R.ia[i+1] ; ++j){
            R.OtherI[j] = i;
        }
    }
    printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n",
           m, n, nnzR);


    f = atoi(argv[3]);
    printf("The latent feature is %i \n", f);

    // create X
    float *X = (float *)calloc(sizeof(float) , m * f);

    // create Y
    float *Y = (float *)malloc(sizeof(float) * n * f);

    for (int i = 0,tot = n*f; i < tot; i++)
       Y[i] = 1;

    // lamda parameter
    float lamda = 0.1;

    // call function

    if (strcmp(method, "als_recsys") == 0)
    {
        als_recsys(&R, X, Y, m, n, f, lamda,nnzR);
    }
    free(X);
    free(Y);
    mtxDestory(&R);
}
/// 13 0.000086 netflix.mtx 8