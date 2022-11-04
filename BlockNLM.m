function u = BlockNLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    R = 5; % radius of search window

    alpha = 2; % radius of block

    n = 1; % distance between the center of each block in image (for non overlapping this value is 2*alpha+1
    f = 1; % distance between the center of each block in window

    add_len = 2*(R+alpha); % additional length

    w0 = extend(u0,m1,m2,add_len);
    w0 = fill(w0,m1,m2,add_len);
    
    w = w0;
    
    temp = zeros(m1+2*add_len,m2+2*add_len);
    entrycount = zeros(m1+2*add_len,m2+2*add_len);
    
    % image traversal

    % stops at last block for which image is covered
    % xsteps = ceil((m1-(2*alpha+1))/n);
    % ysteps = ceil((m2-(2*alpha+1))/n);
    
    % stops at last block that will be in the image
    xsteps = floor(m1/n);
    ysteps = floor(m2/n);

    % stops at last block that will be in the image with first block always centered at (1,1)
    % xsteps = floor((m1+alpha-1)/n);
    % ysteps = floor((m2+alpha-1)/n);

    for ys = 0:ysteps
        for xs = 0:xsteps
            
            % first block is the top left block perfectly fit in the image
            % x = add_len+1+alpha + xs*n;
            % y = add_len+1+alpha + ys*n;

            % first block always centered at top left corner
            x = add_len + 1 + xs*n;
            y = add_len + 1 + ys*n;

            Bi = zeros(2*alpha+1,2*alpha+1);

            % create weights
            sum_weights = 0;
            
            % window traversal
            wsteps = ceil((R-alpha)/f);
            for j = -wsteps:wsteps
                for i = -wsteps:wsteps

                    %coordinates of comparison point
                    x1 = x + i*f;
                    y1 = y + j*f;
                    
                    % comparison block
                    Bj = w(x1-alpha:x1+alpha, y1-alpha:y1+alpha);
                    
                    % Compute squared difference
                    diff = 0;
                    for q = -alpha:alpha
                        for p = -alpha:alpha
                            diff = diff + (w(x+p,y+q)-w(x1+p,y1+q))^2;
                        end
                    end

                    weight = exp(-diff/((2*alpha+1)^2*h^2));

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