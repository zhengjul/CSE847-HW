%% HW5 Q2 part 1: plot given data points. 
x = [0,-1,-3,1,3];
y = [0,2,6,-2,-6];
scatter(x, y);

%% HW5 Q2 part 2: PCA reconstruct images
load USPS.mat

n = 300;
plot = 1;
for j=1:n:3000
    subplot(3,4,plot);
    plot = plot + 1;
    A0 = reshape(A(j,:), 16, 16);
    imshow(A0');
end

% reconstruction error p
p_vals = [10,50,100,200];

% principle_componentA
[principle_component, coordinates] = my_principle_componenta(A);

recons_errs = zeros(numel(p_vals),1);
for i = 1:n:3000
    figure;
    for j = 1:numel(p_vals)
        p = p_vals(j);

        % Reconstruct the images
        pca_image = coordinates(:,1:p) * principle_component(:,1:p)';
        mean_img = mean(A,1);
        for k = 1:size(A,1)  % un-center pca values to backtransform
            pca_image(k,:) = pca_image(k,:) + mean_img;
        end

        % Get the reconstruction errors for each image
        errs = calculate_reconstruction_error(A, pca_image);

        % Reshape the images
        img1 = reshape(pca_image(i,:), 16, 16)';

        % Plot the reconstructed images
        
        prec = 3;
        subplot(1,numel(p_vals),j);
        imshow(img1,[]);
        title(['p = ' num2str(p)]);
        xlabel(['err = ' num2str(errs(1),prec)]);
        
        % Save the reconstruction errors
        recons_errs(j) = sum(errs);
    end
    %saveas(gcf, strcat('q2_',num2str(i,'%d'),'.png'));
end
avg_recons_errs = (recons_errs / size(A,1));

% Print the Reconstruction errors
recons_errs
avg_recons_errs

function [ err ] = calculate_reconstruction_error( original, reconstructed )
% original and reconstructed are n x m matrices where n is the
% number of data points and m is the number of features. 

% Returns a n x 1 vector of errors.

    n = size(original,1);
    diff = original - reconstructed;
    err = zeros(n,1);
    for i = 1:n
        err(i) = norm(diff(i,:),'fro')^2; % Frobenius norm
    end

end
%% principle_componentA
function [principle_component, coordinates] = my_principle_componenta(data)
    [n, d] = size(data);
    % centre data by subtracting the mean of the rows
    centered_data = zeros(size(data));
    for i = 1:d
        centered_data(:,i) = data(:,i) - mean(data(:,i));
    end
    [U, S, V] = svd(centered_data);
    principle_component = V;
    coordinates = U*S;
end
