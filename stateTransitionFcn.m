function predictParticles = stateTransitionFcn(pf, prevParticles,noise)

predictParticles = prevParticles;
predictParticles(:,1) = exp(-prevParticles(:,2)).*prevParticles(:,1);

predictParticles = predictParticles + normrnd(0,noise,size(predictParticles));
    
end