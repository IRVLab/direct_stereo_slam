function [diff_m_ci, diff_m_c, diff_m_i] = processSC(hist, start_idx, end_idx, mask_width)
    hist = hist(start_idx+1:size(hist,1)-end_idx, :);
    hist_t = hist(:,1:size(hist,2)/2);
    hist_i = hist(:,size(hist,2)/2+1:end);

    diff_m_c = process(hist_t, mask_width);
    diff_m_i = process(hist_i, mask_width);

    % weight dist
    w_c = std2(diff_m_c);
    w_i = std2(diff_m_i);
    diff_m_ci = (normalize(diff_m_c,2)*w_c + normalize(diff_m_i,2)*w_i)/(w_c+w_i);
end

function dist = process(hist, mask_width)
    m = size(hist,1);
    for i=1:m
        hist(i,:) = hist(i,:) / norm(hist(i,:));
    end
    dist = inf*ones(m,m);
    for i=1:m
        sig_i = zeros(120, 1200);
        for k=1:60
            sig_i(2*k-1, :) = undist_sc(hist, i, k, false);
            sig_i(2*k, :)   = undist_sc(hist, i, k, true);
        end
        dist_m = 1-sig_i * hist';
        dist_v = min(dist_m);
        dist(i,:) = dist_v;
    end
    for i=1:m
        for j=1:m
            if(abs(i-j)<mask_width)
                dist(i,j) = inf;
            end
        end
    end
end

function sig = undist_sc(hist, i, idx, reverse)
    h_img = reshape(hist(i,:), [20,60]); 
    if(~reverse)
        sig = h_img(:,[idx:1:60,1:1:(idx-1)]);	
    else
        sig = h_img(:,[idx:-1:1,60:-1:(idx+1)]);
    end
    sig = sig(:)';
end