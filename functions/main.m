clearvars;
path(path, 'SLIC_mex');

directoryIm = '../datasets/images/';
dataSet='ambush_2';

I1 = double(imread(fullfile(directoryIm, dataSet, 'frame_0001.png'))) / 256;
I2 = double(imread(fullfile(directoryIm, dataSet, 'frame_0002.png'))) / 256;

directoryOF = '../datasets/flow/';
%Optical flow vector
OFGT = readFlowFile (fullfile(directoryOF, dataSet, 'frame_0001.flo'));

directoryOcc = '../datasets/occlusions/';
%Ground truth occlusion
OccGT = double(imread(fullfile(directoryOcc, dataSet, 'frame_0001.png'))) > 128;

nFig=0;
nFig=nFig+1;
figure(nFig)
imshow(I1);

nFig=nFig+1;
figure(nFig)
imshow(I2);


%% Step 1 and 2: Build xi_1 and eta_12

%Built warping function
I1_from_I2 = warping(I1, I2, OFGT(:,:,1), OFGT(:,:,2));

nFig=nFig+1;
figure(nFig)
imshow(I1_from_I2);

sigma_spatial =1;
sigma_grayLevel =0.1;

eps_g =  1e-3;
sigma = [sigma_spatial, sigma_grayLevel];

[ni, nj, nC] = size(I1);

w = 5;

%Compute the images xi_1 and eta_12 using the cross bilateral filter.
for n=1:nC
    xi_1(:,:,n)  = bfilter2(I1(:,:,n),w,sigma);
    eta_12(:,:,n)  = cross_bilateral_filter(I1(:,:,n), I2(:,:,n),w,sigma);
end

nFig=nFig+1;
figure(nFig)
imshow(xi_1);

nFig=nFig+1;
figure(nFig)
imshow(eta_12);

%% Step 3: Oversegmentation

[ni, nj, nC] = size(xi_1);

weKeep = 0.05; % in percentage
nLabels=round(ni*nj*weKeep/100);

%numSuperpixels is the same as number of superpixels.
[lblP, numSuperpixels] = slicmex(uint8(xi_1*256),nLabels,20);

lblP = double(lblP)+1; %We want labels from 1 to numSuperpixels

%To visualize purpose
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(lblP), hy, 'replicate');
Ix = imfilter(double(lblP), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
idx =gradmag >0;

xi_1_toShow_r = xi_1(:,:,1);
xi_1_toShow_g = xi_1(:,:,2);
xi_1_toShow_b = xi_1(:,:,3);

xi_1_toShow_r(idx) = 1;
xi_1_toShow_g(idx) = 0;
xi_1_toShow_b(idx) = 0;

xi_1_toShow(:,:,1) = xi_1_toShow_r;
xi_1_toShow(:,:,2) = xi_1_toShow_g;
xi_1_toShow(:,:,3) = xi_1_toShow_b;
% end to visualize purpose

nFig=nFig+1;
figure(nFig)
imshow(xi_1_toShow);
%hold on

nFig=nFig+1;
figure(nFig)
imagesc(lblP);
axis off


%% step 4 Gaussian Mixture Estimation

nDist =  2; %Number of gaussians
GM = cell(int16(numSuperpixels),1);

%Fit a gaussian for each superpixel
for n=1:numSuperpixels

    %Select all pixels belonging the same superpixel
     
    
    data =  xi_1(lblP==n);%pixels of xi_1 belonging to superpixel n
    
    try
        %Model the gaussian mixture model of the data
        GM{n} = fitgmdist(data, nDist);
    catch exception
        disp('There was an error fitting the Gaussian mixture model')
        error = exception.message;
        disp(error)
        GM{n} = fitgmdist(data,nDist,'Regularize',0.1);
    end
    %plot3(data(:,1), data(:,2), data(:,3), '*')
    %pause
end

%% step 5 soft-occlusion map

softMap=zeros(ni,nj);
for n=1:numSuperpixels
    
   %Select all pixels belonging the same superpixel
 
    data = eta_12(lblP==n);%pixels of eta_12 belonging to superpixel n
    
    %Probability of belonging the gmm of the superpixel
    postProb = pdf(GM{n}, data);
    
    p=-log(postProb);
    
    softMap(lblP==n)=p;
end

nFig=nFig+1;
figure(nFig)
imagesc(softMap); colorbar

%% step 6: Hard oclsuion map (threshold)

thr =15; %decision threshold
hardMap = softMap>thr; 

nFig=nFig+1;
figure(nFig)
imagesc(hardMap) %Estimated occlusion

%% step 7: Comparison against ground truth

nFig=nFig+1;
figure(nFig)
imagesc(OccGT) %Ground truth occlusion

