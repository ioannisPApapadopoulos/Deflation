%% Self-contained Python script for deflation!
% 
% Finding two solutions of the Bratu equation
%     u''(x) + λeᵘ(x) = 0
%     u(0) = u(1) = 0.
% This problem has one solution if λ = 0 or λ = λ*, 
% two solutions if 0 < λ < λ*, and no solutions if λ > λ* where lmbda* ≈ 3.513830719. 
% 
% Script generated with the aid of ChatGPT from the Python script "bratu_self_contained.py".


n = 100; % number of intervals
h = 1/n; % mesh size
diag_main = -2/h^2 * ones(n, 1);
diag_sub = 1/h^2 * ones(n-1, 1);
Lap = diag(diag_main) + diag(diag_sub, -1) + diag(diag_sub, 1); % finite difference stencil
x = linspace(0, 1, n)';

lmbda = 2.5; % Bratu parameter

% Initial guess
u0 = (1 - x) .* x;

% Run first Newton loop to find first solution
u1 = u0;
for i = 1:100
    u1 = u1 - J(u1, Lap, lmbda)\F(u1, Lap, lmbda);
end

norm(F(u1, Lap, lmbda))

% Run second Newton loop from the SAME initial guess!
u2 = u0;

for i = 1:100
    % Solve origin undeflated Newton system
    du = - J(u2, Lap, lmbda) \ F(u2, Lap, lmbda);
    % Multiply Newton update with the SCALAR tau to 
    % find Newton step of deflated Newton system.
    u2 = u2 + tau(u2, du, u1) * du;
end

norm(F(u2, Lap, lmbda))

% Plot the two different solutions
plot(x, [u1 u2])
%% Functions
% residual + bcs
function r = F(u, Lap, lmbda)
    r = Lap * u + lmbda * exp(u);
    r(1) = 0;
    r(end) = 0;
end

% Jacobian + bcs
function A = J(u, Lap, lmbda)
    A = Lap + lmbda * diag(exp(u));
    A(1,:) = 0; A(:,1) = 0;
    A(end,:) = 0; A(:,end) = 0;
    A(1,1) = 1; A(end,end) = 1;
end

% Hard-coded deflation operator
function m = mu(u, u1) 
    m = 1/(norm(u-u1)^2) + 1;
end

function dm = dmu(u, u1) 
    dm = -2*(u-u1)/norm(u-u1)^4;
end

function t = tau(u, du, u1)
    t = 1 + (1/mu(u, u1) * dot(dmu(u, u1), du)) / (1 - 1/mu(u, u1) * dot(dmu(u, u1), du));
end
