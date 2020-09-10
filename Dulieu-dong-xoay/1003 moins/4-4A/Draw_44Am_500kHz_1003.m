for var = 3:3
    image_real = double(imread(['44Am_500kHz_1003_' num2str(2^var*100) 'u_realpart.tif']));
    image_imag = double(imread(['44Am_500kHz_1003_' num2str(2^var*100) 'u_imagpart.tif']));
    
%     image_real = double(imread(buf_real));
%     image_imag = double(imread(bur_imag));

%     image_real = double(buf_real);
%     image_imag = double(buf_imag);

    
    image_complex = image_real + j*image_imag;
    
    figure()
    mesh(image_real);
    title(['44Am 500kHz 1003 ' num2str(2^var*100) '�m Realpart']);
    
    figure()
    mesh(image_imag);
    title(['44Am 500kHz 1003 ' num2str(2^var*100) '�m Imagpart']);
    
    figure()
    mesh(abs(image_complex));
    title(['44Am 500kHz 1003 ' num2str(2^var*100) '�m Module']);
    
    figure()
    mesh(angle(image_complex));
    title(['44Am 500kHz 1003 ' num2str(2^var*100) '�m Phase']);
end