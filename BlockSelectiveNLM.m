function u = BlockSelectiveNLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    R = 5; % radius of search window

    ssimbound = 0.1; % ssim bound 

    alpha = 2; % radius of block

    n = 2; % distance between the center of each block in image (for non overlapping this value is 2*alpha+1
    f = 1; % distance between the center of each block in window

    add_len = 2*(R+alpha); % additional length

    w0 = extend(u0,m1,m2,add_len);
    w0 = fill(w0,m1,m2,add_len);

    w = w0;
    
    temp = zeros(m1+2*add_len,m2+2*add_len);
    entrycount = zeros(m1+2*add_len,m2+2*add_len);
    
    % image traversal
    y = add_len+1+alpha;
    while y-alpha <= m2+add_len

        x = add_len+1+alpha;
        while x-alpha <= m1+add_len

            Bi = zeros(2*alpha+1,2*alpha+1);

            % create weights
            sum_weights = 0;
            
            % window traversal
            steps = ceil((R-alpha)/f);
            for j = -steps:steps
                for i = -steps:steps

                    %coordinates of comparison point
                    x1 = x + i*f;
                    y1 = y + j*f;
                    
                    % comparison block
                    Bj = w(x1-alpha:x1+alpha, y1-alpha:y1+alpha);

                    %ssim setup
                    k1 = .01;
                    k2 = .03;

                    L = 255;

                    c1 = (k1*L)^2;
                    c2 = (k2*L)^2;

                    mean_A = 0;
                    mean_B = 0;
                    var_A = 0;
                    var_B = 0;

                    cov = 0;
                    
                    % Compute squared difference
                    diff = 0;
                    for q = -alpha:alpha
                        for p = -alpha:alpha
                            diff = diff + (w(x+p,y+q)-w(x1+p,y1+q))^2;

                            %ssim compute mean
                            mean_A = mean_A + w(x+p,y+q);
                            mean_B = mean_B + w(x1+p,y1+q);
                        end
                    end

                    % ssim condition--------------
                    mean_A = mean_A/((2*alpha+1)^2);
                    mean_B = mean_B/((2*alpha+1)^2);

                    % Compute variance
                    for q = -alpha:alpha
                        for p = -alpha:alpha
                            %ssim compute mean
                            var_A = var_A + (w(x+p,y+q)-mean_A)^2;
                            var_B = var_B + (w(x1+p,y1+q)-mean_B)^2;

                            cov = cov + (w(x+p,y+q)-mean_A)*(w(x1+p,y1+q)-mean_B);
                        end
                    end

                    var_A = var_A/((2*alpha+1)^2);
                    var_B = var_B/((2*alpha+1)^2);

                    cov = cov/((2*alpha+1)^2);

                    ssim  = ((2*mean_A*mean_B+c1)*(2*cov+c2))/((mean_A^2+mean_B^2+c1)*(var_A+var_B+c2));
                    
                    weight = 0;

                    if ssim > ssimbound
                        weight = exp(-diff/(h^2));
                    end
                    %-----------------------------
                    
                    Bi = Bi + weight * Bj;

                    sum_weights = sum_weights + weight;
                end
            end
            
            Bi = Bi/sum_weights;

            % apply the weights
            for q = -alpha:alpha
                for p = -alpha:alpha
                    
                    % compute average based on the number of previous entries
                    count = entrycount(x+p,y+q);
                    temp(x+p,y+q) = (count*temp(x+p,y+q) + Bi(alpha+1+p,alpha+1+q))/(count+1);

                    entrycount(x+p,y+q) = count+1;

                end
            end

            x = x + n;
        end

        y = y + n;
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