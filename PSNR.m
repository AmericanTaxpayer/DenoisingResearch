function v = PSNR(A, O)
    [n, m] = size(A);
    M = (255^2).*ones(n, m);
    B = O - A;
    B = B.^2;
    v = 10*log10(sum(M, "all")/sum(B, "all"));
end
