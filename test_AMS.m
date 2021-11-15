x_true = double(imread('barbara.jpg'));
x_true = 0.4*x_true(:, :, 1)+0.4*x_true(:, :, 2)+0.2*x_true(:, :, 3);
b = x_true;
[N, N] = size(x_true);
%%Here we add the noise. We decide what type of noise to add by the menu.
t=menu('Select type of noise:','Gaussian','Impulse','Salt and pepper', 'Gaussian & Impulse','Gaussian & Salt and pepper');
switch t
    case 1
        sigma_gauss=input('Insert noise level for Gaussian noise: ');
        e=randn(size(b));
        e=e/norm(e,'fro')*norm(b,'fro')*sigma_gauss;
        b=b+e;
    case 2
        sigma=input('Insert noise level for impulse noise: ');
        k=round(numel(b)*sigma/2);
        e=randperm(numel(b));
        b(e(1:2*k))=randi(256,1,2*k)-1;
    case 3
        sigma=input('Insert noise level for salt and pepper noise: ');
        k=round(numel(b)*sigma/2);
        e=randperm(numel(b));
        b(e(1:k))=0;
        b(e(k+1:2*k))=255;
    case 4
        sigma_gauss=input('Insert noise level for Gaussian noise: ');
        e=randn(size(b));
        e=e/norm(e,'fro')*norm(b,'fro')*sigma_gauss;
        b=b+e;
        sigma=input('Insert noise level for impulse noise: ');
        k=round(numel(b)*sigma/2);
        e=randperm(numel(b));
        b(e(1:2*k))=randi(256,1,2*k)-1;
    case 5
        sigma_gauss=input('Insert noise level for Gaussian noise: ');
        e=randn(size(b));
        e=e/norm(e,'fro')*norm(b,'fro')*sigma_gauss;
        b=b+e;
        sigma=input('Insert noise level for impulse noise: ');
        k=round(numel(b)*sigma/2);
        e=randperm(numel(b));
        b(e(1:2*k))=randi(256,1,2*k)-1;
end
figure(1)
imshow(reshape(b,N, N), [], 'initialmagnification', 10000, 'border', 'tight')