function u = AdaptiveWindowNLM(u0,h)
    
    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    patch_radius = 2; % radius of comparison patches

    N = 4; % max iterations

    rho = 3.1;

    %lambda = chi2inv(0.99,(2*patch_radius+1)^2-1); % Main adaptive window patch size paper uses df = (2*patch_radius+1)^2
    lambda = 1; % 113.5;

    add_len = 2^(N-1) + patch_radius; % additional length

    w0 = extend(u0,m1,m2,add_len);
    w0 = fill(w0,m1,m2,add_len);

    % first we calculate estiamted standard deviation of noise

    % local residuals
    R = zeros(m1,m2);
    for y = 1:m2
        for x = 1:m1
            R(x,y) = (2*w0(x+add_len,y+add_len)-(w0(x+add_len+1,y+add_len)+w0(x+add_len,y+add_len+1)))/(sqrt(6));
        end
    end

    stdev = 1.4826 * median(abs(abs(R) - median(abs(R),'all')),'all');

    w = zeros(m1+2*add_len,m2+2*add_len,N+1); % nth iteration = w(:,:,n+1)
    v = zeros(m1+2*add_len,m2+2*add_len,N+1); % same size as estimaztion array for sake of conveniance

    w(:,:,1) = w0;
    v(:,:,1) = stdev;

    for n = 1:N
        for y = 1+add_len:m2+add_len
            for x = 1+add_len:m1+add_len
                
                window_radius = 2^(n-1);

                % recalculate weights for new window size

                sum = 0;
                sum_weights = 0;
                sum_square_weights = 0;

                % calulate weights
                for j = -window_radius:window_radius
                    for i = -window_radius:window_radius

                        %coordinates of comparison point
                        x1 = x + i;
                        y1 = y + j;

                        diff1 = 0;
                        diff2 = 0;
                        for q = -patch_radius:patch_radius
                            for p = -patch_radius:patch_radius
                                diff1 = diff1 + ((w(x+p,y+q,n) - w(x1+p,y1+q,n)) / v(x+p,y+q,n))^2;
                                diff2 = diff2 + ((w(x+p,y+q,n) - w(x1+p,y1+q,n)) / v(x1+p,y1+q,n))^2;
                            end
                        end

                        diff = double((diff1+diff2)/2);

                        weight = exp(-diff/(lambda*(h^2)));
                        
                        sum = sum + weight * w(x1,y1,1);
                        sum_weights = sum_weights + weight;
                        sum_square_weights = sum_square_weights + weight^2;
                    end
                end

                w(x,y,n+1) = sum / sum_weights;
                v(x,y,n+1) = stdev * (sqrt(sum_square_weights)) / sum_weights;
                
            end
        end

        v(:,:,n+1) = fill(v(:,:,n+1),m1,m2,add_len);
        w(:,:,n+1) = fill(w(:,:,n+1),m1,m2,add_len);
    end

    u = zeros(m1,m2);

    % now that values have been solved up to N iterations we pick the best estimator for each pixel
    for y = 1:m2
        for x = 1:m1

            % coordinated of point in extended space to adjust for padding in w
            x1 = x + add_len;
            y1 = y + add_len;
            
            % find max index n for which |\hat{u}_{i,n}-\hat{u}_{i,k}| < \hat{v}_{i,k}
            max_index = 1; % initialize for the statistically improbable case that this condition is not satisfied for any n

            for n = 2:N  % first iteration is not checked

                bool = true;

                % check variational condition
                for k = 1:n-1
                    if abs(w(x1,y1,n+1)-w(x1,y1,k+1)) >= rho*v(x1,y1,k+1)

                        bool = false;
                        break;
                    end
                end

                if bool
                    max_index = n;
                end
            end

            u(x,y) = w(x1,y1,max_index+1);
        end
    end

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
end