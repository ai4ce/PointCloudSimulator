clear;close all;

%% settings
env_idx = 'v0';
pose_idx = 'pose0';

n_pose_on_traj = 256;
angle = 10; % +- angle between two poses
angle = angle/180*pi;
displacement = 5;
%% 
data_dir =  fullfile('../results/',env_idx);
save_dir = fullfile(data_dir,[env_idx,'_',pose_idx]);

if exist(save_dir,'dir')~=7
    mkdir(save_dir)
end

img_name =fullfile(data_dir,'environment.png');
I = imread(img_name);
figure;imshow(I);hold on;
[xi,yi] = getline;

%%
% sample location
locations = samplePointsOnPolyline([xi,yi],n_pose_on_traj);
% slight move the location around the sampled locations
locations = locations + displacement*randn(n_pose_on_traj,2);

% sample orientation
theta = zeros(n_pose_on_traj,1);
theta(1) = rand(1)*2*pi - pi;
for ii = 2:n_pose_on_traj
    theta(ii) = theta(ii-1) + rand(1)*2*angle-angle;
end

% visualize
pose = [locations,theta];
plot(xi,yi);
% plot(locations(:,1),locations(:,2),'ro');
quiver(locations(:,1),locations(:,2),cos(theta),sin(theta) ,'ro','showarrowhead','on','autoscalefactor',0.5);
print(fullfile(save_dir,'gt_pose'),'-dpng');
close()
% 

save_name = fullfile(save_dir,'gt_pose.mat');
save(save_name,'pose');
