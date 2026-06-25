array A[400];
array B[400];
array C[400, 400];

for i0 in 0 .. 400 {
    read A[i0];
    write B[i0];
    for i1 in 0 .. i0 {
        read B[i0];
        read C[i0, i1];
        read B[i1];
        write B[i0];
    }
    read B[i0];
    read C[i0, i0];
    write B[i0];
}
