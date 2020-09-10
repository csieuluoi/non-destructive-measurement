image_module = [1 2 3 4 5 6 7 8 9
    2 2 2 2 2 2 2 2 2
    3 3 3 3 3 3 3 3 3];

max = zeros(1,8);
max(1,1) = image_module(1,9);

max_test = image_module(1,1);

image_size = size(image_module);

for num = 1:7
    for lin = 1:image_size(1)
        for col = 1:image_size(2)
            if (image_module(lin,col) >= max_test)
                for kk = 1:num
                    buf(kk) = abs(image_module(lin,col) - max(1,kk));
                    disp(kk);
                    %disp(buf(kk));
                end
                
                flag = 1;
                
                for kk = 1:num
                    flag = flag && buf(kk);
                end
                
                if flag ~= 0
                    max_test = image_module(lin,col);
                    max(1,num+1) = max_test;
                end
            end
        end
       
    end
    max_test = image_module(1,1);
end

max
ave_max = mean(max)