array A[256];
array B[256];
array C[256, 256];
array D[256];
array E[256, 256];

for i0 in 0 .. 256 {
    write A[i0];
    write B[i0];
    for i1 in 0 .. 256 {
        read A[i0];
        read C[i0, i1];
        read D[i1];
        write A[i0];
        read B[i0];
        read E[i0, i1];
        read D[i1];
        write B[i0];
    }
    read A[i0];
    read B[i0];
    write B[i0];
}
