function u = NLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    R = 5;
    r = 2;

    add_len = R+r; % additional length

    w = extend(u0,m1,m2,add_len);
    w = fill(w,m1,m2,add_len);
    
    temp = zeros(m1+2*add_len,m2+2*add_len);
    
    % Non-Local Means Algorithm

    % Matlab traverses 2D arrays column-wise
    for y = add_len+1:m2+add_len
        for x = add_len+1:m1+add_len

            % create and apply weights
            sum = 0;
            sum_weights = 0;
            
            for j = -R:R
                for i = -R:R

                    %coordinates of comparison point
                    x1 = x + i;
                    y1 = y + j;

                    diff = 0; % squared difference of neighborhoods centered at (x,y) and (x1,y1)
                    for q = -r:r
                        for p = -r:r
                            diff = diff + (w(x+p,y+q)-w(x1+p,y1+q))^2;
                        end
                    end

                    weight = exp(-diff/((2*r+1)^2*h^2));

                    sum = sum + weight * w(x1,y1);

                    sum_weights = sum_weights + weight;
                end
            end

            temp(x,y) = sum/sum_weights;

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