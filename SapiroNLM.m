function u = SapiroNLM(u0,h)

    u0 = double(u0);
    [m1,m2] = size(u0); % we compute the boubnds once and then pass it to the subfunctions to save computations

    R = 5;
    r = 2;

    errbound = .1;
    gradbound = 6;
    thetabound = .5;

    add_len = R + r; % additional length

    w = extend(u0,m1,m2,add_len);
    w = fill(w,m1,m2,add_len);
    
    temp = zeros(m1+2*add_len,m2+2*add_len);
    
    % Non-Local Means Algorithm

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
                    
                    % sum of the windows centered at (x,y) and (x1,y1)
                    sum1 = 0;
                    sum2 = 0;

                    diff = 0; % squared difference of neighborhoods centered at (x,y) and (x1,y1)
                    
                    Dx1 = 0; Dx2 = 0;
                    Dy1 = 0; Dy2 = 0;

                    for q = -r:r
                        for p = -r:r
                            sum1 = sum1 + w(x+p,y+q);
                            sum2 = sum2 + w(x1+p,y1+q);

                            diff = diff + (w(x+p,y+q)-w(x1+p,y1+q))^2;

                            if p < r
                                Dx1 = Dx1 + (w(x+p+1,y+q)-w(x+p,y+q))^2;
                                Dx2 = Dx2 + (w(x1+p+1,y1+q)-w(x1+p,y1+q))^2;
                            end

                            if q < r
                                Dy1 = Dy1 + (w(x+p,y+q+1)-w(x+p,y+q))^2;
                                Dy2 = Dy1 + (w(x1+p,y1+q+1)-w(x1+p,y1+q))^2;
                            end
                        end
                    end

                    if sum1 < sum2
                        err = abs(1-(sum1/sum2));
                    else
                        err = abs(1-(sum2/sum1));
                    end

                    Dx1 = Dx1 / (2*r*(2*r+1)); Dx2 = Dx2 / (2*r*(2*r+1));
                    Dy1 = Dy1 / (2*r*(2*r+1)); Dy2 = Dy2 / (2*r*(2*r+1));

                    avgrad1 = sqrt(Dx1^2+Dy1^2);
                    avgrad2 = sqrt(Dx2^2+Dy2^2);

                    costheta = (Dx1*Dx2+Dy1*Dy2)/(avgrad1*avgrad2);

                    weight = 0;

                    if err < errbound
                        if avgrad1 < gradbound || avgrad2 < gradbound || 1 - costheta < thetabound
                            weight = exp(-diff/((2*r+1)^2*h^2));
                        end
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