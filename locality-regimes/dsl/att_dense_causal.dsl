params n, d;
array Q[n, d];
array K[n, d];
array V[n, d];
array S[n, n];
array P[n, n];
array O[n, d];
array M[n];

for i in 0 .. n {
    for j in 0 .. i + 1 {
        for k in 0 .. d {
            read Q[i, k];
            read K[j, k];
            write S[i, j];
        }
    }
}
for i in 0 .. n {
    for j in 0 .. i + 1 {
        read S[i, j];
        write M[i];
    }
}
for i in 0 .. n {
    for j in 0 .. i + 1 {
        read S[i, j];
        read M[i];
        write P[i, j];
    }
}
for i in 0 .. n {
    for j in 0 .. i + 1 {
        for k in 0 .. d {
            read P[i, j];
            read V[j, k];
            write O[i, k];
        }
    }
}
