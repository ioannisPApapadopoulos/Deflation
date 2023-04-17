% Finding two solutions of the Bratu equation
%     u''(x) + λeᵘ(x) = 0
%     u(0) = u(1) = 0.
% This problem has one solution if λ = 0 or λ = λ*, 
% two solutions if 0 < λ < λ*, and no solutions if λ > λ* where λ* ≈ 3.513830719. 

addpath(genpath('../'));
lmbda = 2.5; % Bratu parameter

n = 100; % number of intervals
h = 1/n; % mesh size
D = spdiags([1/h^2*ones(n,1), -2/h^2*ones(n,1), 1/h^2*ones(n,1)], [-1, 0, 1], n, n); % finite difference stencil

x = linspace(0, 1, n);

% Initial guess
u0 = (1-x).*x;

% Initialise deflation operator, currently holding no known solutions. 
% Should probably use mass matrix for the inner product, but we simply 
% pass the identity for now.
deflation = DeflationOperator({}, speye(length(x)));

% Initialise the solver, and pass the deflation operator into it
nls = NonlinearSolver('deflation', deflation);

% residual and Jacobian
F = @(u) residual(u, lmbda, D);
J = @(u) jacobian(u, lmbda, D);

% Use Newton to find the first solution
u1 = nls.newton(u0, F, J);
norm(F(u1))

% Update list of known solutions
nls.deflation.updateFoundSolutions(u1);

% Rerun Newton from the same initial guess to find second solution.
u2 = nls.newton(u0, F, J);
norm(F(u2))

% Plot solutions
plot(x, u1)
hold on
plot(x, u2)

% residual + bcs
function r = residual(u, lmbda, D)
    r = D*u + lmbda*exp(u);
    r(1) = 0; r(end) = 0;
end

% Jacobian + bcs
function J = jacobian(u, lmbda, D)
    J = D + lmbda*diag(exp(u));
    J(:,1) = 0; J(1,:) = 0;
    J(1,1) = 1; J(end,end) = 1;
end
