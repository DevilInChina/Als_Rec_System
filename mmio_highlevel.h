#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif

#include "mmio.h"
#include "baseOpt.h"
#include <sys/mman.h>

#define MAX_BUFFER_SIZE 1024ll*1024*1024*2
//#include "common.h"
enum MtxType{
    Pattern,Real,Complex,Integer
};
// read matrix infomation from mtx file
int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename)
{
    int m_tmp, n_tmp, nnz_mtx_report;
    MM_typecode matcode;
    /* find out size of sparse matrix .... */
    FILE*f;
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    int ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_mtx_report;

    return 0;
}

// read matrix infomation from mtx file
int mmio_data(int *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrVal, char *filename)
{
    int m_tmp, n_tmp, nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report,isSymmetric_tmp=0;
    int MtxType=-1;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { MtxType = Pattern; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { MtxType = Real; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { MtxType = Complex; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { MtxType = Integer; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
    }
    else
    {

    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;
        switch (MtxType){
            case Real:
                returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
                break;
            case Complex:
                returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
                break;
            case Integer:
                returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
                fval = ival;
                break;
            case Pattern:
                returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
                fval = 1.0;
                break;
            default:
                printf("Error In Read.");
                return 0;
        }
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
       // printf("%%%f\n",100.0*i/nnz_mtx_report);
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
double readDouble(char **buffer){
    while (!isdigit(**buffer)){
        ++*buffer;
    }

    int flag = 1;
    if(*(*buffer-1)=='-') {
        flag = -flag;
    }
    int ret = 0,dep = 0,dec = 1;
    while (isdigit(**buffer)){
        ret = (ret<<1)+(ret<<3)+**buffer-'0';
        ++*buffer;
    }
    if(**buffer=='.'){
        ++*buffer;
        while (isdigit(**buffer)){
            dep = (dep<<1)+(dep<<3)+**buffer-'0';
            ++*buffer;
            dec = (dec<<1)+(dec<<3);
        }
    }
    return flag*(ret+1.0*dep/dec);
}
int readInt(char **buffer){

    while (!isdigit(**buffer)){
        ++*buffer;
    }

    int flag = 1;
    if(*(*buffer-1)=='-') {
        flag = -flag;
    }
    int ret = 0;
    while (isdigit(**buffer)){
        ret = (ret<<1)+(ret<<3)+**buffer-'0';
        ++*buffer;
    }
    return ret*flag;
}
int mmap_io_data(int *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrVal, char *filename) {
    int m_tmp, n_tmp, nnz_tmp;
    int fd = 0;
    if ((fd = open(filename, O_RDONLY)) < 0) {
        printf("error\n");
    }

    char *fp = mmap(NULL, MAX_BUFFER_SIZE, PROT_READ, MAP_SHARED, fd, 0);

    if (fp == NULL) {
        printf("error in mmap\n");
        return 1;
    }

    unsigned totSz = sizeof(char) * (strlen(fp) + 1);

    char *bufferdIn = (char *) malloc(totSz);
    if (bufferdIn == NULL) {
        printf("Cannot alloc memory\n");
        return 1;
    }
    memcpy(bufferdIn, fp, totSz);

    char *bufferIter = bufferdIn;

    char buffer[512] = {0};

    do {
        sscanf(bufferIter, "%[^\n]", buffer);
        bufferIter += strlen(buffer) + 1;
    } while (buffer[0] == '%');

    sscanf(buffer, "%d %d %d", &m_tmp, &n_tmp, &nnz_tmp);

    int *csrRowPtr_counter = (int *) calloc(sizeof(int),(m_tmp + 1));
    int *csrRowIdx_tmp = (int *) malloc(nnz_tmp * sizeof(int));
    int *csrColIdx_tmp = (int *) malloc(nnz_tmp * sizeof(int));
    VALUE_TYPE *csrVal_tmp = (VALUE_TYPE *) malloc(nnz_tmp * sizeof(VALUE_TYPE));


    //printf("%d %d %d\n", m_tmp, n_tmp, nnz_tmp);
    int cnt =0;
    for (int i = 0; i < nnz_tmp; i++) {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;
        idxi = readInt(&bufferIter);
        idxj = readInt(&bufferIter);
        fval = readDouble(&bufferIter);

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        ++csrRowPtr_counter[idxi+1];
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
        //if(100.0*i/nnz_tmp>10)return 0;
    }

    for(int i = 0; i < m_tmp ; ++i){
        csrRowPtr_counter[i+1]+=csrRowPtr_counter[i];
    }


    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

    for (int i = 0; i < nnz_tmp; i++) {
        int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
        csrColIdx[offset] = csrColIdx_tmp[i];
        csrVal[offset] = csrVal_tmp[i];
        csrRowPtr_counter[csrRowIdx_tmp[i]]++;
    }
    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);
    free(bufferdIn);
    return 0;
}
#endif
