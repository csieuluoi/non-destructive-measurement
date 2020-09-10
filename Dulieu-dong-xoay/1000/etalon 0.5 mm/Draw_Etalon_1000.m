for var = 1:3
    image_real = double(imread(['Etalon_1000_' num2str(2^var*100) 'u_realpart.tif']));
    image_imag = double(imread(['Etalon_1000_' num2str(2^var*100) 'u_imagpart.tif']));
    
%     image_real = double(imread(buf_real));
%     image_imag = double(imread(bur_imag));

%     image_real = double(buf_real);
%     image_imag = double(buf_imag);

    
    image_complex = image_real + j*image_imag;
    
    figure()
    mesh(image_real);
    title(['Etalon 1000 (500kHz) ' num2str(2^var*100) 'µm Realpart']);
    
    figure()
    mesh(image_imag);
    title(['Etalon 1000 (500kHz) ' num2str(2^var*100) 'µm Imagpart']);
    
    figure()
    mesh(abs(image_complex));
    title(['Etalon 1000 (500kHz) ' num2str(2^var*100) 'µm Module']);
    
    figure()
    mesh(angle(image_complex));
    title(['Etalon 1000 (500kHz) ' num2str(2^var*100) 'µm Phase']);
    
%     figure()
%     plot(image_complex-mean(mean(image_complex))*ones);
%     title(['2E 400kHz ' num2str(2^var*100) 'µm Phase']);
end