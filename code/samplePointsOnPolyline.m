function points = samplePointsOnPolyline(xy,nPoints)
% xy: Nx2 matrix representing line segments. [x,y]
% nPoints: number of points to be sampled
% reference:
%   https://blogs.mathworks.com/steve/2012/07/06/walking-along-a-path/

d = diff(xy,1);
dist_from_v_to_v = hypot(d(:,1),d(:,2));
cum_dist_along_path = [0;
    cumsum(dist_from_v_to_v,1)];

dist_steps = linspace(0,cum_dist_along_path(end),nPoints);
points = interp1(cum_dist_along_path,xy,dist_steps);


end
