for var = 1:3
    buf_real = imread(['2A_500kHz_' num2str(2^var*100) 'u_realpart.tif']);
    buf_imag = imread(['2A_500kHz_' num2str(2^var*100) 'u_imagpart.tif']);
    
%     image_real = double(imread(buf_real));
%     image_imag = double(imread(bur_imag));

    image_real = double(buf_real);
    image_imag = double(buf_imag);

    
    image_complex = image_real + j*image_imag;
    
    figure()
    mesh(image_real);
    title(['2A 500kHz ' num2str(2^var*100) '�m Realpart']);
    
    figure()
    mesh(image_imag);
    title(['2A 500kHz ' num2str(2^var*100) '�m Imagpart']);
    
    figure()
    mesh(abs(image_complex));
    title(['2A 500kHz ' num2str(2^var*100) '�m Module']);
    
    figure()
    mesh(angle(image_complex));
    title(['2A 500kHz ' num2str(2^var*100) '�m Phase']);
end

% Image Processing: delta_max calculation
% Fissure depth = 400�m and 800�m

%=========== 200�m =========================
%-- Original matrix size (81,25)
% ----- Cut: 10 first lines, 10 last lines -------
% ----- Cut: 3 first columns, 1 last column ----
for var = 1:1
    image_real = double(imread(['2A_500kHz_' num2str(2^var*100) 'u_realpart.tif']));
    image_imag = double(imread(['2A_500kHz_' num2str(2^var*100) 'u_imagpart.tif']));
    
    image_complex = image_real + j*image_imag;
    
    image_module = abs(image_complex);
    image_size =size(image_module);

% Calculation of image background:
line_top = zeros(1,image_size(2)-4);
line_bottom = line_top;

    for col = 4:image_size(2)-1
        for line = 11:25
            line_top(:,col-3) = line_top(:,col-3)+ image_module(line,col);
        end
        
        for line = image_size(1)-24:image_size(1)-10
            line_bottom(:,col-3) = line_bottom(:,col-3) + image_module(line,col);
        end
        
    end
    
    line_top = line_top/15;
    line_bottom = line_bottom/15;
    line_average = 0.5*(line_top+line_bottom);

% Lap ma tran anh nen:

    for line = 1:image_size(1)-20
        image_ground(line,:) = line_average;
    end

% Hieu chinh kich thuoc anh cu cho phu hop voi anh nen:
    for line = 1:image_size(1)-20
        for col = 4:image_size(2)-1
            image_corr(line,col-3) = image_module(line+10,col);
        end
    end

% Ma tran anh bo nen:
    %     image_sub =  image_ground - image_corr;
    image_sub =  abs(image_ground - image_corr);
    figure()
    mesh(image_sub);
    title(['2A 500kHz ' num2str(2^var*100) '�m -- Subtracted Module']);

% Tim gia tri lon nhat so voi nen va toa do cua diem anh tuong ung:
    im_corr_size = size(image_corr);

    delta_max = (image_sub(1,1));
    max_lin = 1;
    max_col = 1;

    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if image_sub(line,col) - delta_max > 0
                delta_max = image_sub(line,col);
                max_lin = line;
                max_col = col;
            end
        end
    end
    
    %------------------------------------------------------
    %*********--- Tinh goc pha cua diem max ---*********** 
    image_phase = angle(image_complex);
    angle_max = image_phase(max_lin+10,max_col+3);
    
    figure()
    plot(image_complex(max_lin+10,:),'b-'), axis('equal')
%     plot(image_complex,'b-'), axis('equal')
    hold on
    plot(image_complex(max_lin+10,max_col+3),'ro')
    
    %***********--- Tim chieu dai/rong cua khe ho ---*************
    for line = 1:20
        noise_mat(line,:) = image_sub(line,:);
    end
    
    limit = 6*mean(std(noise_mat)); %6 ///////////////////////////////
    
    length_finding = zeros(im_corr_size(1),im_corr_size(2));
    
    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if image_sub(line,col) >= limit
                length_finding(line,col) = image_sub(line,col);
            end
        end
    end
    
    %-- Tim chieu dai --
    mark = 0;
    
    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if (length_finding(line,col) > 0) && (mark == 0)
                first_line = line;
                mark = 1;
            elseif (length_finding(line,col) > 0) && (mark == 1)
                last_line = line;
            end
        end
    end
    
    %-- Tim chieu rong --
    mark = 0;
    
%     for col = 1:im_corr_size(2)
%         for line = 1:im_corr_size(1)
%             if (length_finding(line,col) > 0) && (mark == 0)
%                 first_col = col;
%                 mark = 1;
%             elseif (length_finding(line,col) > 0) && (mark == 1)
%                 last_col = col;
%             end
%         end
%     end
    
%     for col = 1:im_corr_size(2)
%         for line = 1:im_corr_size(1)
%             if (length_finding(line,col) > 0.1*limit) && (mark == 0)
%                 first_col = col;
%                 mark = 1;
%             elseif (length_finding(line,col) > 0) && (mark == 1)
%                 last_col = col;
%             end
%         end
%     end
    
    mark1 = 0;
%     for col = 1:im_corr_size(2)
%           if (mean(length_finding(:,col)) == 0) && (mark1 == 0) && (col > first_col)
%               first_last_col = col - 1;
%               mark1 = 1;
%           end
%     end
    
    for col = 1:im_corr_size(2)
          if (mean(length_finding(:,col)) > 0.05*limit) && (mark == 0)
              first_col = col;
              mark = 1;
          end
    end

    for col = 1:im_corr_size(2)
          if (mean(length_finding(:,col)) < 0.1*limit) && (mark1 == 0) && (col > first_col)
              first_last_col = col - 1;
              mark1 = 1;
          end
    end
    %----  ----  ----
    
    figure()
    mesh(length_finding)
    title(['2A 500kHz ' num2str(2^var*100) '�m -- Length Finding']);
    
    %*******--- Module tim 8 gia tri lon nhat ---*******
    max = zeros(1,8);
    max(1,1) = length_finding(max_lin,max_col);

    max_test = length_finding(1,1);
    
    for num = 1:7
        for lin = 1:im_corr_size(1)
            for col = 1:im_corr_size(2)
                if (length_finding(lin,col) >= max_test)
                    for kk = 1:num
                        buf(kk) = abs(length_finding(lin,col) - max(1,kk));
                    end
                
                    flag = 1;
                
                    for kk = 1:num
                        flag = flag && buf(kk);
                    end
                
                    if flag ~= 0
                        max_test = length_finding(lin,col);
                        max(1,num+1) = max_test;
                    end
                end
            end
       
        end
        max_test = length_finding(1,1);
    end
   
    % ----**** Cach 2 tim GTTB ****----
    for lin = max_lin-4:max_lin+4
        for col = max_col-2:max_col+2
            max_mat(lin - (max_lin - 5),col - (max_col - 3)) = length_finding(lin,col);
        end
    end
    %------------------------------------------------------
    
    
    disp(['2A 500kHz ' num2str(2^var*100) '�m'])
    disp(['delta_max = ' num2str(delta_max)])
%     delta_max
    disp('Position of the max point in the new co-ordinate:')
    disp(['Line = ' num2str(max_lin) '   Col = ' num2str(max_col)])
%     max_lin
%     max_col
    
    disp('Position of the max point in the original co-ordinate:')
    disp(['Line = ' num2str(max_lin + 10) '   Col = ' num2str(max_col + 3)])
    disp('')


    disp(['Angle of max point is ' num2str((angle_max*180)/pi) ' degrees'])
    
    disp(['Length of fissure is: ' num2str(last_line - first_line) ' steps'])
    
    %*****--- Hien gia tri max trung binh ---*****
    disp(['8 biggest values of image_sub matrix are: ' num2str(max)])
    disp(['Average value of 8 biggest values: AVE_delta_max = ' num2str(mean(max))])
    %--------
    disp(['AVE_delta_max1_' num2str(2^var*100) '�m = ' num2str(mean(mean(max_mat)))])
    
    disp(['Width of fissure is ' num2str(first_last_col - first_col) ' steps'])
    disp('*********-----------*********')
    
end

%=========== 400�m   and  800�m ============ 
%---- Cut: 19 first columns, 1 last column ----
%---- Cut: 3 first lines, 3 last lines     ----
for var = 2:3
    image_real = double(imread(['2A_500kHz_' num2str(2^var*100) 'u_realpart.tif']));
    image_imag = double(imread(['2A_500kHz_' num2str(2^var*100) 'u_imagpart.tif']));
    
    image_complex = image_real + j*image_imag;
    
    image_module = abs(image_complex);
    image_size =size(image_module);

% Calculation of image background:
line_top = zeros(1,image_size(2)-20);
line_bottom = line_top;

    for col = 20:image_size(2)-1
        for line = 4:15
            line_top(:,col-19) = line_top(:,col-19)+ image_module(line,col);
        end
        
        for line = image_size(1)-14:image_size(1)-3
            line_bottom(:,col-19) = line_bottom(:,col-19) + image_module(line,col);
        end
        
    end
    
    line_top = line_top/12;
    line_bottom = line_bottom/12;
    line_average1 = 0.5*(line_top+line_bottom);

% Lap ma tran anh nen:

    for line = 1:image_size(1)-6
        image_ground1(line,:) = line_average1;
    end

% Hieu chinh kich thuoc anh cu cho phu hop voi anh nen:
    for line = 1:image_size(1)-6
        for col = 20:image_size(2)-1
            image_corr1(line,col-19) = image_module(line+3,col);
        end
    end

% Ma tran anh bo nen:
    %     image_sub1 =  image_ground1 - image_corr1;
    image_sub1 =  abs(image_ground1 - image_corr1);
    figure()
    mesh(image_sub1);
    title(['2A 500kHz ' num2str(2^var*100) '�m -- Subtracted Module']);

% Tim gia tri lon nhat so voi nen va toa do cua diem anh tuong ung:
    im_corr_size = size(image_corr1);

    delta_max = (image_sub1(1,1));
    max_lin = 1;
    max_col = 1;

    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if image_sub1(line,col) - delta_max > 0
                delta_max = image_sub1(line,col);
                max_lin = line;
                max_col = col;
            end
        end
    end
    
    %---------------------------------------------------
    image_phase = angle(image_complex);
    angle_max = image_phase(max_lin+3,max_col+19);
    
    figure()
    plot(image_complex(max_lin+3,:),'b-'), axis('equal')
%     plot(image_complex,'b-'), axis('equal')
    hold on
    plot(image_complex(max_lin+3,max_col+19),'ro')
    
    %**********--- Tim chieu dai cua khe ho ---***********
    for col = 1:20
        noise_mat1(:,col) = image_sub1(:,col);
    end
    
    limit1 = 7*mean(std(noise_mat));    %7 //////////////////////////////
    
    length_finding1 = zeros(im_corr_size(1),im_corr_size(2));
    
    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if image_sub1(line,col)>=limit1
                length_finding1(line,col) = image_sub1(line,col);
            end
        end
    end
    
    mark = 0;
    
    for line = 1:im_corr_size(1)
        for col = 1:im_corr_size(2)
            if (length_finding1(line,col) > 0) && (mark == 0)
                first_line = line;
                mark = 1;
            elseif (length_finding1(line,col) > 0) && (mark == 1)
                last_line = line;
            end
        end
    end
    
    %-- Tim chieu rong --
    mark = 0;
    
%     for col = 1:im_corr_size(2)
%         for line = 1:im_corr_size(1)
%             if (length_finding1(line,col) > 0) && (mark == 0)
%                 first_col = col;
%                 mark = 1;
%             elseif (length_finding1(line,col) > 0) && (mark == 1)
%                 last_col = col;
%             end
%         end
%     end
    
%     for col = 1:im_corr_size(2)
%         for line = 1:im_corr_size(1)
%             if (length_finding1(line,col) > 0.1*limit1) && (mark == 0)
%                 first_col = col;
%                 mark = 1;
%             elseif (length_finding1(line,col) > 0) && (mark == 1)
%                 last_col = col;
%             end
%         end
%     end
    
    mark1 = 0;
%     for col = 1:im_corr_size(2)
%           if (mean(length_finding1(:,col)) == 0) && (mark1 == 0) && (col > first_col)
%               first_last_col = col - 1;
%               mark1 = 1;
%           end
%     end

    for col = 1:im_corr_size(2)
          if (mean(length_finding1(:,col)) > 0.05*limit1) && (mark == 0)
              first_col = col;
              mark = 1;
          end
    end
    
    for col = 1:im_corr_size(2)
          if (mean(length_finding1(:,col)) < 0.1*limit1) && (mark1 == 0) && (col > first_col)
              first_last_col = col - 1;
              mark1 = 1;
          end
    end
    %----  ----  ----
    
    figure()
    mesh(length_finding1)
    title(['2A 500kHz ' num2str(2^var*100) '�m -- Length Finding']);
    
    %*******--- Module tim 8 gia tri lon nhat ---*******
    max = zeros(1,8);
    max(1,1) = length_finding1(max_lin,max_col);

    max_test = length_finding1(1,1);
    
    for num = 1:7
        for lin = 1:im_corr_size(1)
            for col = 1:im_corr_size(2)
                if (length_finding1(lin,col) >= max_test)
                    for kk = 1:num
                        buf(kk) = abs(length_finding1(lin,col) - max(1,kk));
                    end
                
                    flag = 1;
                
                    for kk = 1:num
                        flag = flag && buf(kk);
                    end
                
                    if flag ~= 0
                        max_test = length_finding1(lin,col);
                        max(1,num+1) = max_test;
                    end
                end
            end
       
        end
        max_test = length_finding1(1,1);
    end
    
    % ----**** Cach 2 tim GTTB ****----
    for lin = max_lin-4:max_lin+4
        for col = max_col-2:max_col+2
            max_mat(lin - (max_lin - 5),col - (max_col - 3)) = length_finding1(lin,col);
        end
    end
   
    %---------------------------------------------------
    
    disp(['2A 500kHz ' num2str(2^var*100) '�m'])
    disp(['delta_max = ' num2str(delta_max)])
%     delta_max
    disp('Position of the max point in the new co-ordinate:')
    disp(['Line = ' num2str(max_lin) '   Col = ' num2str(max_col)])
%     max_lin
%     max_col
    
    disp('Position of the max point in the original co-ordinate:')
    disp(['Line = ' num2str(max_lin + 3) '   Col = ' num2str(max_col + 19)])
    disp('')
%     max_lin + 3
%     max_col + 19

    disp(['Angle of max point is ' num2str((angle_max*180)/pi) ' degrees'])
    
    disp(['Length of fissure is: ' num2str(last_line - first_line) ' steps'])
    
    %*****--- Hien gia tri max trung binh ---*****
    disp(['8 biggest values of image_sub matrix are: ' num2str(max)])
    disp(['Average value of 8 biggest values: AVE_delta_max = ' num2str(mean(max))])
    
    %--------
    disp(['AVE_delta_max1_' num2str(2^var*100) '�m = ' num2str(mean(mean(max_mat)))])
    
    disp(['Width of fissure is ' num2str(first_last_col - first_col) ' steps'])
    disp('*********-----------*********')
    
end
