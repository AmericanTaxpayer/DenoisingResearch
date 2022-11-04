function [u,v,w] = BlockAdaptiveWindowNLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    N = 4; % max number of iterations

    alpha = 2; % radius of block

    rho = 3;

    %lambda = chi2inv(0.99,(2*patch_radius+1)^2-1); % Main adaptive window patch size paper uses df = (2*patch_radius+1)^2

    % CHANGE THIS DEPENDING IN BLOCK SIZE
    lambda = 113.5;
    % CHANGE THIS DEPENDING IN BLOCK SIZE

    g = 1; % distance between the center of each block in image (for this algorith to work with Adaptive Window it is required that g=1) 
    f = 1; % distance between the center of each block in window

    add_len = 2^N + 2*alpha; % additional length

    w0 = extend(u0,m1,m2,add_len);
    w0 = fill(w0,m1,m2,add_len);

    % local residuals
    R = zeros(m1,m2);
    for y = 1:m2
        for x = 1:m1
            R(x,y) = (2*w0(x+add_len,y+add_len)-(w0(x+add_len+1,y+add_len)+w0(x+add_len,y+add_len+1)))/(sqrt(6));
        end
    end

    stdev = 1.4826 * median(abs(abs(R) - median(abs(R),'all')),'all');
    
    w = zeros(m1+2*add_len,m2+2*add_len,N+1);
    v = zeros(m1+2*add_len,m2+2*add_len,N+1); % same size as estimaztion array for sake of conveniance

    w(:,:,1) = w0;
    v(:,:,1) = stdev;
    
    % image traversal

    % stops at last block for which image is covered
    % xsteps = ceil((m1-(2*alpha+1))/n);
    % ysteps = ceil((m2-(2*alpha+1))/n);

    % stops at last block that will be in the image with first block always centered at (1,1)
    % xsteps = floor((m1+alpha-1)/n);
    % ysteps = floor((m2+alpha-1)/n);

    blocks = zeros(2*alpha+1,2*alpha+1,m1,m2,N);

    for n = 1:N

        window_radius = 2^(n-1) + alpha; % + alpha ensures blocks fit within window;

        temp = zeros(m1+2*add_len,m2+2*add_len);
        entrycount = zeros(m1+2*add_len,m2+2*add_len);

        for y = 1+add_len:m2+add_len
            for x = 1+add_len:m2+add_len

                sum_weights = 0;
                sum_square_weights = 0;

                B = zeros(2*alpha+1,2*alpha+1);

                % create weights
                sum_weights = 0;
            
                % window traversal
                for j = -window_radius:window_radius
                    for i = -window_radius:window_radius

                        %coordinates of comparison point
                        x1 = x + i;
                        y1 = y + j;
                    
                        % comparison block
                        B1 = w(x1-alpha:x1+alpha,y1-alpha:y1+alpha,n);
                    
                        % Compute squared difference
                        diff1 = 0;
                        diff2 = 0;
                        for q = -alpha:alpha
                            for p = -alpha:alpha
                                diff1 = diff1 + ((w(x+p,y+q,n) - w(x1+p,y1+q,n))/v(x,y,n))^2;
                                diff2 = diff2 + ((w(x+p,y+q,n) - w(x1+p,y1+q,n))/v(x1,y1,n))^2;
                            end
                        end

                        diff = (diff1+diff2)/2;

                        weight = exp(-diff/(lambda*(h^2)));

                        B = B + (weight*B1);

                        sum_weights = sum_weights + weight;
                        sum_square_weights = sum_square_weights + weight^2;
                    end
                end
            
                B = B / sum_weights;

                blocks(:,:,x-add_len,y-add_len,n) = B;

                % apply the weights
                for q = -alpha:alpha
                    for p = -alpha:alpha

                        % compute average based on the number of previous entries
                        count = entrycount(x+p,y+q);
                        temp(x+p,y+q) = (count*temp(x+p,y+q) + B(alpha+1+p,alpha+1+q))/(count+1);

                        entrycount(x+p,y+q) = count+1;

                    end
                end

                v(x,y,n+1) = stdev * (sqrt(sum_square_weights)) / sum_weights;
            end
        end

        w(:,:,n+1) = fill(temp,m1,m2,add_len);
        v(:,:,n+1) = fill(v(:,:,n+1),m1,m2,add_len);
        
    end

    u = zeros(m1,m2);
    
    temp = zeros(m1+2*add_len,m2+2*add_len);
    entrycount = zeros(m1+2*add_len,m2+2*add_len);
    
    for y = 1:m2
        for x = 1:m1

                % coordinated of point in extended space to adjust for padding in w and v
                x1 = x + add_len;
                y1 = y + add_len;

                % find max index n for which |\hat{u}_{i,n}-\hat{u}_{i,k}| < \hat{v}_{i,k}
                max_index = 1; % initialize for the statistically improbable case that this condition is not satisfied for any n
                for n = 2:N

                    bool = true;

                    % check variational condition
                    for k = 1:n-1
                    
                        dev = 0;
                        devbound = 0;
                        for q = -alpha:alpha
                            for p = -alpha:alpha
                                dev = dev + abs(blocks(alpha+1+p,alpha+1+q,x,y,n)-blocks(alpha+1+p,alpha+1+q,x,y,k));

                                devbound = devbound + v(x1+p,y1+q,k);
                            end
                        end

                            
                        if dev >= rho*devbound

                            bool = false;
                            break;
                        end
                    end

                    if bool
                        max_index = n;
                    end
                end

                B = blocks(:,:,x,y,max_index);

                % apply the weights
                for q = -alpha:alpha
                    for p = -alpha:alpha

                        % compute average based on the number of previous entries
                        count = entrycount(x1+p,y1+q);
                        temp(x1+p,y1+q) = (count*temp(x1+p,y1+q) + B(alpha+1+p,alpha+1+q))/(count+1);

                        entrycount(x1+p,y1+q) = count+1;

                    end
                end
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