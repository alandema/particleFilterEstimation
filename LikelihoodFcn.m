function likelihood = LikelihoodFcn(pf, particles, measurement, Sigma)
% pfBlockMeasurementLikelihoodFcnExample Measurement likelihood function for particle filter
%
% The measurement is the first state
%
% Inputs:
%    particles   - A NumberOfStates-by-NumberOfParticles matrix
%    measurement - System output, a scalar
%
% Outputs:
%    likelihood - A vector with NumberOfParticles elements whose n-th
%                 element is the likelihood of the n-th particle

%#codegen

% % Predicted measurement
% yHat = particles(:,1);
% 
% % Calculate likelihood of each particle based on the error between actual
% % and predicted measurement
% %
% % Assume error is distributed per multivariate normal distribution with
% % mean value of zero, variance 1. Evaluate the corresponding probability
% % density function
% e = bsxfun(@minus, yHat, measurement); % Error
% numberOfMeasurements = 1;
% mu = 0; % Mean
% Sigma = eye(numberOfMeasurements); % Variance
% measurementErrorProd = dot((e-mu), Sigma \ (e-mu), 2);
% c = 1/sqrt((2*pi)^numberOfMeasurements * det(Sigma));
% likelihood = c * exp(-0.5 * measurementErrorProd);


dx = (particles(:,1) - measurement).^ 2;
likelihood = (1/sqrt((2*pi*Sigma)))* exp(-dx / (2.0 * Sigma));
end

