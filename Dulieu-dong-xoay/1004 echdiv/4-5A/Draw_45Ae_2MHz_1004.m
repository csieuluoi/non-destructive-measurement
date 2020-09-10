for var = 3:3
    image_real = double(imread(['45Ae_2MHz_' num2str(2^var*100) 'u_realpart.tif']));
    image_imag = double(imread(['45Ae_2MHz_' num2str(2^var*100) 'u_imagpart.tif']));
    
%     image_real = double(imread(buf_real));
%     image_imag = double(imread(bur_imag));

%     image_real = double(buf_real);
%     image_imag = double(buf_imag);

    
    image_complex = image_real + j*image_imag;
    
    figure()
    mesh(image_real);
    title(['45Ae 2MHz 1004 ' num2str(2^var*100) 'µm Realpart']);
    
    figure()
    mesh(image_imag);
    title(['45Ae 2MHz 1004 ' num2str(2^var*100) 'µm Imagpart']);
    
    figure()
    mesh(abs(image_complex));
    title(['45Ae 2MHz 1004 ' num2str(2^var*100) 'µm Module']);
    
    figure()
    mesh(angle(image_complex));
    title(['45Ae 2MHz 1004 ' num2str(2^var*100) 'µm Phase']);
end