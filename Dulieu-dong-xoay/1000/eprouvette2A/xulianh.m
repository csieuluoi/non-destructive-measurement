image = [2 1 1 1 1 1 1 1 2
    3 1 1 1 1 1 1 1 2
    2 1 1 1 5 1 1 1 2
    2 1 1 1 1 1 1 1 2
    3 1 1 1 1 1 1 1 5];

image_size =size(image);

% Tinh toan nen anh, bo qua ria anh:

for col = 2:image_size(2)-1
    line_top(:,col-1) = 0.5*(image(1,col)+image(2,col));
    line_bottom(:,col-1) = 0.5*(image(image_size(1)-1,col)+image(image_size(1),col));
    line_average(:,col-1) = 0.5*(line_top(:,col-1)+line_bottom(:,col-1));
end

% Lap ma tran anh nen:

for line = 1:image_size(1)
    image_ground(line,:) = line_average;
end

% Hieu chinh kich thuoc anh cu cho phu hop voi anh nen:
for line = 1:image_size(1)
    for col = 2:image_size(2)-1
        image_corr(line,col-1) = image(line,col);
    end
end

% Ma tran anh bo nen:
image_sub = image_corr - image_ground;

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
% Chay dung se cho gia tri delta_max = 4
% Toa do cua gia tri max la (3,4)
delta_max
max_lin
max_col
