function [I2_warped] = warping(I1, I2, u, v)
%Inputs:
    %I1, I2, input images
    %u, v: vectors optical flow

%Output:
    %I2_warped: image warped I1 from I2

[M, N, C] = size(I1);

%Take the pixel positions of I1 and apply u,v, taking into account the
%borders

idxx = (1:N)+u;
idyy = (1:M)'+v;

% idxx(idxx < 1) = 1;
% idyy(idyy < 1) = 1;
% idxx(idxx > M) = M;
% idyy(idyy > N) = N;

%warp the image in each channel
for i=1:C
    I2_warped(:,:,i) = interp2(I2(:,:,i),idxx,idyy,'cubic');
end
