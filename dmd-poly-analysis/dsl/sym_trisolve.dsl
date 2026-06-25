params p0, A_d0, B_d0, C_d0, C_d1;
array A[A_d0];
array B[B_d0];
array C[C_d0, C_d1];

for i0 in 0 .. p0 {
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
