array A[390];
array B[390];
array C[410];
array D[410, 390];
array E[410];

for i0 in 0 .. 410 {
    write A[i0];
    for i1 in 0 .. 390 {
        read B[i1];
        read C[i0];
        read D[i0, i1];
        write B[i1];
        read A[i0];
        read E[i1];
        write A[i0];
    }
}
