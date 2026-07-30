params n, d;
array Q[n, d];
array K[n, d];
array V[n, d];
array S[d, d];
array O[n, d];

for i in 0 .. n {
    for a in 0 .. d {
        for b in 0 .. d {
            read K[i, a];
            read V[i, b];
            read S[a, b];
            write S[a, b];
        }
    }
    for a in 0 .. d {
        for b in 0 .. d {
            read Q[i, a];
            read S[a, b];
            write O[i, b];
        }
    }
}
