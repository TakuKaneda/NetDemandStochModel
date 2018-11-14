# Pkg.add("Distributions")
# using DataFrames, JuMP, Gurobi, CSV
using DataFrames, CSV
using Distributions, StatsFuns
using Plots
include("src/source.jl")
include("bound_data.jl")  # data of upper/lower bounds

nodes_df = Read_nodes_csv("data/multilayer-nodes.csv")  # see src/source.jl
Nodes = nodes_df[:NodeID]
# Node2Layer = nodes_df[:Node2Layer]
## network data
d = length(Nodes); cov = 0.6; n = 5000;
H =24; NLayers = 50;
n_lattice = 5;
## distribution data
# for beta distribution
a = 6;b = 6;
# for ARMA model
phi = 0.6; theta = 0.8;

SIGMA = ones(d,d)*cov + eye(d).*(1-cov);
# println("#Nodes: ", d,", #Layers: ",NLayers);
## copula & ARMA(1,1) model
Phi = phi*ones(d); Theta = theta*ones(d); # parameter for ARMA
x = zeros(H,d,n);
mvn = MvNormal(zeros(d),SIGMA);  # defnine multivariate normal
# Z = rand(mvn, n); # sampling
x[1,:,:] = rand(mvn, n);
err_prev = x[1,:,:];
for t = 2:H
    err = rand(mvn, n); # spacially dependent noise
    x[t,:,:] = Phi.*x[t-1,:,:] + err + Theta.*err_prev;
    err_prev = err;  # update the error term
end

## Take the inverse of x, (denote u)
# normalization
x_mean = reshape(mean(x,3), Val{2});
x_std = reshape(std(x,3), Val{2});
z = (x.-x_mean)./x_std;
clear!(:x);
u = normcdf.(z);  # convert by the cdf of standard normal
clear!(:z);

## plot u
# pyplot()
# histogram(u[10,1,:])
# pyplot()
# scatter(u[10,1,:],u[10,2,:])

## convert to beta function
betadist = Beta(a, b); # define the beta distribution
@time betaval = quantile.(betadist, u);
clear!(:u);

## shift to the true value
WIDTH = (UB - LB);
t = 1:H;
# ND = zeros(H,d,n);
ND = betaval.*WIDTH .+ LB;
clear!(:betaval);
## plot the result
# nd = 100; # node
# pyplot()
# plot(ND[:,nd,1:10])
# plot!(mean(ND[:,nd,:],2))
# plot!(LB[:,nd])
# plot!(UB[:,nd])
