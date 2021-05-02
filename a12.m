function a = a12(x, y)
    %m and n are the two sets of encoding values
    [p,h,s] = ranksum(x,y);
    
    [m,tmp] = size(x);
    [n,tmp] = size(y);
    a = (s.ranksum/50 - (50+1)/2)/50;
    if a < 0.50,
        [p,h,s] = ranksum(y,x);
        a = (s.ranksum/50 - (50+1)/2)/50;
        a = -1 * a;
    end
end
