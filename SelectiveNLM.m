function u = SelectiveNLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the bounds once and then pass it to the subfunctions to save computations

    R = 4;
    r = 2;

    ssimbound = -.2;
    bound1 = .25;
    bound2 = .006;

    add_len = R+r; % additional length

    w = extend(u0,m1,m2,add_len);
    w = fill(w,m1,m2,add_len);
    
    temp = zeros(m1+2*add_len,m2+2*add_len);

    max_mean = 0;
    max_var = 0;
    means = zeros(m1+2*add_len,m2+2*add_len);
    vars = zeros(m2+2*add_len,m2+2*add_len);

    % pre-processing
    for y = 1+add_len:m2+add_len
        for x = 1+add_len:m1+add_len

                    mean = 0;
                    var = 0;

                    %compute mean
                    for q = -r:r
                        for p = -r:r
                            mean = mean + w(x+p,y+q);
                        end
                    end
                    
                    mean = mean/((2*r+1)^2);

                    % Compute variance
                    for q = -r:r
                        for p = -r:r
                            var = var + (w(x+p,y+q)-mean)^2;
                        end
                    end

                    var = var/((2*r+1)^2);

                    means(x,y) = mean;
                    vars(x,y) = var;

                    if mean > max_mean
                        max_mean = mean;
                    end

                    if var > max_var
                        max_var = var;
                    end
        end
    end

    means = fill(means,m1,m2,add_len);
    vars = fill(vars,m1,m2,add_len);
    
    % Non-Local Means Algorithm

    for y = 1+add_len:m2+add_len
        for x = 1+add_len:m1+add_len

            % create and apply weights
            sum = 0;
            sum_weights = 0;

            for j = -R:R
                for i = -R:R

                    %coordinates of comparison point
                    x1 = x + i;
                    y1 = y + j;

                    %ssim setup
                    mean = means(x,y);
                    var = vars(x,y);
                    mean1 = means(x1,y1);
                    var1 = vars(x1,y1);

                    cov = 0;

                    diff = 0; % squared difference of neighborhoods centered at (x,y) and (x1,y1)
                    for q = -r:r
                        for p = -r:r
                            diff = diff + (w(x+p,y+q)-w(x1+p,y1+q))^2;

                            cov = cov + (w(x+p,y+q)-mean)*(w(x1+p,y1+q)-mean1);
                        end
                    end

                    cov = cov/((2*r+1)^2);

                    % compute adaptive weighted ssim
                    k1 = .01;
                    k2 = .03;

                    L = 255;

                    c1 = (k1*L)^2;
                    c2 = (k2*L)^2;
                    c3 = c2 / 2;

                    lum = (2*mean*mean1+c1) / (mean^2+mean1^2+c1);
                    cont = (2*sqrt(var*var1)+c2) / (var+var1+c2);
                    struc = (cov+c3) / (sqrt(var*var1)+c3);

                    % test for high variance
                    if var > bound1 * max_var
                        lum = nthroot(lum^3,4);
                        cont = nthroot(cont^3,4);
                        struc = sqrt(abs(struc^3)) * (struc/abs(struc));
                    end

                    % test for low variance regions
                    if var < bound2 * max_var
                        lum = sqrt(lum^3);
                        cont = nthroot(cont^3,4);
                        struc = nthroot(abs(struc^3),4) * (struc/abs(struc));
                    end

                    ssim = lum * cont * struc;
                    
                    weight = 0;

                    if ssim > ssimbound
                        weight = exp(-diff/((2*r+1)^2*h^2));
                    end

                    sum = sum + weight * w(x1,y1);
                    sum_weights = sum_weights + weight;
                end
            end
            
            temp(x,y) = sum / sum_weights;

        end
    end

    u = trim(temp,m1,m2,add_len);

    function A = fill(a,m1,m2,add_len)
        % a: m1+2*add_len x m2+2*add_len
    
        for p = 1:add_len
            a(add_len+1-p,:) = a(add_len+1+p,:);

            a(m1+add_len+p,:) = a(m1+add_len-p,:);
        end

        for q = 1:add_len
            a(:,add_len+1-q) = a(:,add_len+1+q);

            a(:,m2+add_len+q) = a(:,m2+add_len-q);
        end

        A = a;
    end

    function A = trim(a,m1,m2,add_len)
        % A: m1+2*add_len x m2+2*add_len
        ret = zeros(m1,m2);

        for q = 1:m2
          for p = 1:m1
             ret(p,q) = a(p+add_len,q+add_len);
          end
        end

        A = ret;
    end

    function A = extend(a,m1,m2,add_len)
        % a: m1 x m2

        ret = zeros(m1+2*add_len,m2+2*add_len);

        for q = add_len+1:m2+add_len
            for p = add_len+1:m1+add_len
                ret(p,q) = a(p-add_len,q-add_len);
            end
        end

        A = ret;
    end
end