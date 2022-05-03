clear
rng(1)

measuData = [1 0.9351 0.8512 0.9028 0.7754 0.7114 0.6830 0.6147 0.5628 0.7090];

pf = robotics.ParticleFilter;
pf.StateTransitionFcn =@stateTransitionFcn;
pf.MeasurementLikelihoodFcn =@LikelihoodFcn;
pf.StateEstimationMethod = 'mean';
pf.ResamplingMethod = 'systematic';

initialize(pf, 6500, [0.9,1.2;0,0.05]);

sigma = 0.01;
noise = 0.005;

state =[];

for k=1:10 
    [PredictedState,PredictedCovariance]= predict(pf, noise);
    [CorrectedState,CorrectedCov] = correct(pf,measuData(k),sigma);
    state(:,k) = pf.getStateEstimate;
end

for i=11:20
    [PredictedState,PredictedCovariance]= predict(pf, noise);
    state(:,i) = pf.getStateEstimate;
end

figure
scatter(1:10,measuData)
hold on
plot(1:20,state(1,:),'-*')
scatter(20,0.3)
ylim([0,1.2])
legend('measu','estimate')
grid on
