image_complex=image_complex(2:end,:);
nx = size(image_complex,1); ny = size(image_complex,2);
Dx = -diag(ones(nx-1,1),1)+eye(nx);Dx(nx,nx)=0;Dx=kron(eye(ny),Dx);
figure(22),imagesc(real(image_complex)),colorbar,title('real')
figure(21),imagesc(imag(image_complex)),colorbar,title('imag')
figure(20),imagesc(abs(image_complex))
figure(20),imagesc(abs(image_complex)),title('abs')

Dy = -diag(ones(ny-1,1),1)+eye(ny);Dy(ny,ny)=0;Dy=kron(Dy,eye(nx));
Derx=abs(Dx*image_complex(:)*1).^2+abs(Dy*image_complex(:)*0).^2;
Dery=abs(Dx*image_complex(:)*0).^2+abs(Dy*image_complex(:)*1).^2;
figure(23),imagesc(reshape((Derx),nx,ny)),colorbar
figure(24),imagesc(reshape((Dery),nx,ny)),colorbar,title('der y')

DerRI=(Dx*real(image_complex(:))).^2+(Dy*real(image_complex(:))).^2;
figure(25),mesh(reshape((DerRI),nx,ny)),colorbar,title('der real I')
figure(27),mesh(reshape(abs(image_complex),nx,ny)),colorbar,title(' abs I')
figure(25),mesh(reshape((DerRI).^(-1),nx,ny)),colorbar,title('der real I')
figure(25),mesh(reshape((DerRI),nx,ny)),colorbar,title('der real I')