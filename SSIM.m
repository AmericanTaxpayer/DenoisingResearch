function ssim = SSIM(u,v)

    k1 = .01;
    k2 = .03;

    L = 255;

    c1 = (k1*L)^2;
    c2 = (k2*L)^2;

    [m,n] = size(u);

    mean_u = 0;
    mean_v = 0;
    var_u = 0;
    var_v = 0;

    cov = 0;

    for y = 1:n
        for x = 1:m
            mean_u = mean_u + u(x,y);
            mean_v = mean_v + v(x,y);
        end
    end

    mean_u = mean_u/(m*n);
    mean_v = mean_v/(m*n);

    for y = 1:n
        for x = 1:m
            var_u = var_u + (u(x,y)-mean_u)^2;
            var_v = var_v + (v(x,y)-mean_v)^2;

            cov = cov + (u(x,y)-mean_u)*(v(x,y)-mean_v);
        end
    end

    var_u = var_u/(m*n);
    var_v = var_v/(m*n);
    cov = cov/(m*n);

    ssim  = ((2*mean_u*mean_v+c1)*(2*cov+c2))/((mean_u^2+mean_v^2+c1)*(var_u+var_v+c2));
end