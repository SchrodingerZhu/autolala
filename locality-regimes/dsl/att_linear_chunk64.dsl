params T, d, QT, KT, VT, OT;
array Q[QT, d];
array K[KT, d];
array V[VT, d];
array A[64, 64];
array S[d, d];
array O[OT, d];

for c in 0 .. T {
    for i in 0 .. 64 {
        for j in 0 .. 64 {
            for k in 0 .. d {
                read Q[64 * c + i, k];
                read K[64 * c + j, k];
                write A[i, j];
            }
        }
    }
    for i in 0 .. 64 {
        for j in 0 .. 64 {
            for k in 0 .. d {
                read A[i, j];
                read V[64 * c + j, k];
                write O[64 * c + i, k];
            }
        }
    }
    for i in 0 .. 64 {
        for a in 0 .. d {
            for b in 0 .. d {
                read Q[64 * c + i, a];
                read S[a, b];
                write O[64 * c + i, b];
            }
        }
    }
    for i in 0 .. 64 {
        for a in 0 .. d {
            for b in 0 .. d {
                read K[64 * c + i, a];
                read V[64 * c + i, b];
                read S[a, b];
                write S[a, b];
            }
        }
    }
}
