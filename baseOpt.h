
#define BASE_OPT
#ifdef BASE_OPT
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <sys/time.h>

#define OFAST

#define CG_SWITCH

//#define SPEC_MATMAT_TRANSB_MULTITHREAD

//#define UPDATE_MTX_MULTITHREAD

#define CORE_NUMBER 160

#define THREAD_NUMBERS 160
#define type float




typedef struct sparseMtx{
    int m,n;
    int *ia;/// m+1
    int *ja;/// nnz
    int *OtherI; /// nnz
    type *val;
}sparseMtx;

#define pos(i,j,M) ((i)*(M)+(j))

///@todo add parallel on matmat
#define trans_Next(i,m,n)\
((i)%(n)*(m))+((i)/(n))

#define trans_Prev(i,m,n)\
((i)%(m)*(n))+((i)/(m))

typedef struct Para{
    sparseMtx *Mtx;
    float *Unchange;
    float *Update;
    int f;
    float lamda;
    int begL;
    int endL;
    double time_prepareA;
    double time_prepareb;
    double time_solver;
}Para;

#ifdef OFAST

//void cg(float *A, float *x, float *b, int n, int *iter, int maxiter, float threshold) __attribute__((optimize("Ofast")));
#endif
inline type dotprod(type *res,const type *a,const type *b,int len) {
    (*res) = 0;
    for (const type *ba = a, *bb = b, *ea = a + len; ba != ea; ++ba, ++bb)(*res) += *ba * (*bb);
}
void transMoveData(type *mtx,int i,int m,int n){
    type s = mtx[i];
    int cur = i;
    int pre = trans_Prev(i,m,n);
    while (pre!=i){
        mtx[cur]=mtx[pre];
        cur = pre;
        pre = trans_Prev(cur,m,n);
    }
    mtx[cur] = s;
}

void trans_to_T_matrix(type*A,int m,int n){
    //  MALLOC(tempA,type,m*n);
    int k = m*n;
    for(int i = 0 ; i < k ; ++i){
        int next = trans_Next(i,m,n);
        while (next>i){
            next = trans_Next(next,m,n);
        }
        if(next==i){
            transMoveData(A,i,m,n);
        }
    }
}


void matmat(type *C,const type *A,type *B,int m,int k,int n){
    //trans_to_T_matrix(B,k,n);

    memset(C,0, sizeof(type)*m*n);
//#pragma omp parallel for
    for(int i = 0 ; i < m ; ++i){
        for(int j = 0 ; j < n ; ++j){
            for(int kk = 0 ; kk < k ; ++kk){

                C[pos(i,j,n)]+=A[pos(i,kk,k)]*B[pos(kk,j,n)];
            }
        }
    }
    //trans_to_T_matrix(B,n,k);
}
double kkk = 0;
void matmat_BtB(float *C, const float *BT, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    type res;

    for (int i = 0; i < m; i++) {
        for (int j = i; j < n; j++) {
            dotprod(&res, BT + i * k, BT + j * k, k);
            C[pos(i,j,n)] = res;
            if(i!=j){
                C[pos(j,i,m)] = res;
            }
        }
    }
}

int check(type *a,int len){
    int cnt = 0;
    type *be = a+len;
    //int i = 0;
    while (a!=be){
        if(isnan(*a)) {
            ++cnt;
            //           printf("%d\n",i);
            //         fflush(stdout);
        }
        ++a;
    }
    return cnt;
}
///void specMatmat_transB(const sparseMtx*cp,type*ret,type *A,type*BT,int m,int k,int n) __attribute__((optimize("Ofast")));

typedef struct TransB_Parameter{
    const sparseMtx*cp;
    type *ret;
    type *A;
    type *BT;
    int m,k,n,begL,endL;
}TransB_Parameter;
#ifdef SPEC_MATMAT_TRANSB_MULTITHREAD
void specMatMat_transB_part(TransB_Parameter*para){
//#pragma omp parallel for
    for(int i = para->begL ; i < para->endL; ++i){
        dotprod(para->ret+i,para->A+para->cp->OtherI[i]*para->k,para->BT+para->cp->ja[i]*para->k,para->k);
        para->ret[i] = para->cp->val[i]-para->ret[i];
        para->ret[i]*=para->ret[i];
    }
}
void specMatmat_transB(const sparseMtx*cp,type*ret,type *A,type*BT,int m,int k,int n){
    int nnz = cp->ia[m] - cp->ia[0];
    int kk =  (nnz)%CORE_NUMBER;
    int totThreads = CORE_NUMBER+(kk!=0);
    TransB_Parameter *para=malloc(sizeof(TransB_Parameter)*(totThreads));
    pthread_t *pids = malloc(sizeof(pthread_t)*totThreads);
    int block = (nnz)/CORE_NUMBER;
    for(int i = 0 ; i < totThreads ; ++i){
        para[i].cp = cp;
        para[i].A = A;
        para[i].BT = BT;
        para[i].begL = i*block;
        para[i].endL = i!=CORE_NUMBER?((i + 1) * block):nnz;
        para[i].ret=ret;
        para[i].m=m;
        para[i].n=n;
        para[i].k=k;
    }
    for(int i = 0 ; i < totThreads ; ++i){
        pthread_create(pids+i,NULL,(void*)specMatMat_transB_part,para+i);
    }
    for(int i = 0 ; i < totThreads; ++i){
        pthread_join(pids[i],NULL);
    }
    free(para);
    free(pids);
}
#else
void specMatmat_transB(const sparseMtx*cp,type*ret,type *A,type*BT,int m,int k,int n){
    int nnz = cp->ia[m] - cp->ia[0];
#pragma omp parallel for
    for(int i = 0 ; i < nnz; ++i){
        //printf("%d\n",cp->OtherI[i]);
        dotprod(ret+i,A+cp->OtherI[i]*k,BT+cp->ja[i]*k,k);
        ret[i] = cp->val[i]-ret[i];
        ret[i]*=ret[i];
        //exit(0);
    }
}
#endif


float dotproduct(float *vec1, float *vec2, int n)
{
    float result =0;
    dotprod(&result,vec1,vec2,n);
    return result;
}

void matvec(float *A, float *x, float *y, int m, int n)
{
///#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        dotprod(y+i,A+i*n,x,n);
    }
}

float vec2norm(float *x, int n)
{
    float sum = 0;
    dotprod(&sum,x,x,n);
    return sqrtf(sum);
}

void residual(type *A,type*x,type *b,type *y,int n){
    for(int i = 0 ; i < n ; ++i)y[i] = 0.0;
    matvec(A,x,y,n,n);
    for(int i = 0 ; i < n ; ++i)y[i] = b[i] - y[i];
}
type getF(type *a,type *A,int n,type *temp){////using temp storage
    type ret = 0;
    memset(temp,0, sizeof(type)*n);
    matvec(A,a,temp,n,n);
    dotprod(&ret,a,temp,n);
    return ret;
}
void showMtx(const type *a,int n,int m){
    printf("[");
    for(int i = 0 ; i < n ; ++i){
        printf("[");
        for(int j = 0 ; j < m ; ++j){
            printf("%7.9f",*(a+i*m+j));
            if(j!=m-1)printf(",");
        }
        printf("]");
        if(i!=n-1)printf(",\n");
    }
    printf("]\n");
}


int getnnz(type* A,int tot){
    type *e = A+tot;
    int ret = 0;
    while (A!=e){
        if(fabsf(*A)>0.1)++ret;
        ++A;
    }
    return ret;
}
double cgbegT;

void check_empMem(void *p){
    if(p==NULL){
        printf("memory alloc failed.\n");
        exit(0);
    }else{
        printf("memory checked succeed.\n");
        fflush(stdout);
    }
}


#ifdef CG_SWITCH
void cg(float *A, float *x, float *b, int n, int *iter, int maxiter, float threshold) {
    memset(x, 0, sizeof(float) * n);
    float *residual = (float *) malloc(sizeof(float) * n);
    float *y = (float *) malloc(sizeof(float) * n);
    float *p = (float *) malloc(sizeof(float) * n);
    float *q = (float *) malloc(sizeof(float) * n);
    *iter = 0;
    float norm = 0;
    float rho = 0;
    float rho_1 = 0;

    // p0 = r0 = b - Ax0
    matvec(A, x, y, n, n);
    for (int i = 0; i < n; i++)
        residual[i] = b[i] - y[i];
    type vecB = vec2norm(b, n) * threshold;
    do {
        //printf("\ncg iter = %i\n", *iter);
        rho = dotproduct(residual, residual, n);
        if (*iter == 0) {
            memcpy(p, residual, sizeof(type) * n);
        } else {
            float beta = rho / rho_1;
            for (int i = 0; i < n; i++) {
                p[i] = residual[i] + beta * p[i];
            }
        }

        matvec(A, p, q, n, n);
        float alpha = rho / dotproduct(p, q, n);

        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];

        for (int i = 0; i < n; i++)
            residual[i] -= alpha * q[i];

        rho_1 = rho;
        float error = vec2norm(residual, n);

        *iter += 1;

        if (error < vecB)
            break;
    } while (*iter < maxiter);

    free(residual);
    free(y);
    free(p);
    free(q);
}
#else
void cg(type *A, type *x, type *b, int n, int *iter, int maxiter, type threshold) {
    struct timeval t1,t2;
    gettimeofday(&t1,NULL);
    type *p = malloc(sizeof(type) * n);
    memset(x,0, sizeof(type)*n);
    residual(A, x, b, p, n);
    type *r = malloc(sizeof(type) * n);
    memcpy(r, p, sizeof(type) * n);
    type *temp = (type*)malloc(sizeof(type)*n);
    for (int i = 0; i < n; ++i)x[i] = 0.0;
    type rdot, rdot_1 = 0;
    type bvecMM = 0;
    dotprod(&bvecMM,b,b,n);
    bvecMM = sqrtf(bvecMM)*threshold;
    dotprod(&rdot, r, r, n);
    for (*iter = 0; *iter < maxiter; ++*iter) {

        type beta;
        if (!*iter);
        else {
            beta = rdot / rdot_1;
            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * p[i];
            }
        }
        type alpha = getF(p, A, n,temp);
        //showMtx(&alpha,1,1);
        alpha = rdot / alpha;

        for (int i = 0; i < n; ++i) {
            x[i] += p[i] * alpha;
            r[i] -= temp[i] * alpha;
        }
        rdot_1 = rdot;
        dotprod(&rdot, r, r, n);
        if(sqrtf(rdot)<bvecMM)break;
    }

    free(temp);
    free(r);
    free(p);
    gettimeofday(&t2,NULL);
    cgbegT+=(t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
}
#endif

void mtxDestory(sparseMtx *P){
    free(P->val);
    free(P->ja);
    free(P->ia);
    free(P->OtherI);
}

void transpose(float *AT, const float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}


void csrToCsc(const sparseMtx*Mtx,sparseMtx *ret){
    ret->n = Mtx->n;
    ret->m = Mtx->m;
    ret->ia = calloc(sizeof(int),(ret->n+1));
    ret->ja = malloc(sizeof(int)*(Mtx->ia[Mtx->m] - Mtx->ia[0]));
    ret->OtherI = malloc(sizeof(int)*(Mtx->ia[Mtx->m] - Mtx->ia[0]));
    ret->val = malloc(sizeof(type)*(Mtx->ia[Mtx->m] - Mtx->ia[0]));
    int m = ret->m,n = ret->n;

    int *cnt = (int*)calloc(sizeof(int),(n+1));
    int nnz = Mtx->ia[m] - Mtx->ia[0];
    int row = 0,col;

    for(int i = 0 ; i < nnz ; ++i){
        ret->ia[Mtx->ja[i]+1]++;
    }

    for(int i = 1 ; i <= n ; ++i) {
        ret->ia[i] += ret->ia[i - 1];
    }

    for(int i = 0 ; i < nnz ; ++i){
        col = Mtx->ja[i];
        ret->ja[ret->ia[col]+cnt[col]] = row;
        ret->val[ret->ia[col]+cnt[col]] = Mtx->val[i];
        ++cnt[col];
        Mtx->OtherI[i] = row;
        if(i+1==Mtx->ia[row+1]){
            ++row;
        }
    }

    free(cnt);
}

#endif
/*
0 1 3 5 ia
1 0 2 0 1  ja
1 2 1 9 11 val

 0 1  0
 2 0  1
 9 11 0
 */
