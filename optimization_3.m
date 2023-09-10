%-------------------------------------------------------------------------%
%
%                        Convex Optimization 
%
%                           PROJECT 3
%
%                   Toganidis Nikos - 2018030085
%                Tzimpimpaki Evaggelia - 2018030165
%
%-------------------------------------------------------------------------%

clear; close all; clc; %#ok<CLALL>


%% Excercise 1 and 2 - 2D DRAWINGS
  
  Center = [1,1];
  Radius = 2;
  
  x0 = [3, 3];
  x_interior = [0,0];  
  
  %  ----------- CHANGE THIS BASED ON THE EXERCISE -----------
  % Exercise 1 
  % x_star = Radius /norm(x0) * x0;

  % Exercise 2 
   x_star = Center + Radius * (x0 - Center) /norm(x0 - Center);  
   
  % Create and plot ball and projections  
  angle = linspace(0,2*pi); 
  x = Radius*cos(angle) + Center(1);
  y = Radius*sin(angle) + Center(2);
  
  figure(); plot(Center(1)+x,Center(2)+y);
  fill(x,y,'m'); axis([-5 5 -3 6]); 
  hold on; plot(Center(1),Center(2),'.k');  
  plot(x0(1),x0(2),'xk');
  plot(x_star(1),x_star(2),'*k');  
  plot(x_interior(1),x_interior(2),'xr');  
  plot(x_interior(1),x_interior(2),'*r');  
  line([x0(1), x_star(1)], [x0(2), x_star(2)], 'color', 'b');
  hold off; xlabel('X'); ylabel('Y'); title('Problem schema in 2D');
  legend('Ball', 'Center (1,1)', 'Point x0 = [3,3]', 'Projection of x0 onto the Ball', ...,
      'Point x1 = [0,0]','Projection of x1 onto the Ball','Location', 'NorthWest');

  

%% Excercise 1 and 2 - 3D DRAWINGS
  
  Center = [1,1,1];
  x0 = [-3, 2.5, 3];  
  
  %  ----------- CHANGE THIS BASED ON THE EXERCISE -----------
  % Exercise 1 
  % x_star = Radius /norm(x0) * x0;

  % Exercise 2 
   x_star = Center + Radius * (x0 - Center) /norm(x0 - Center);  

  % Create and plot ball and projections  
  figure(); [X,Y,Z] = sphere(50);
  surf(Radius*X+Center(1),Radius*Y+Center(2),Radius*Z+Center(3));
  hold on; plot3(Center(1),Center(2),Center(3),'.k');
  plot3(x0(1),x0(2),x0(3),'xk'); axis square;
  plot3(x_star(1),x_star(2),x_star(3),'*k');
  line([x0(1),x_star(1)],[x0(2),x_star(2)],[x0(3),x_star(3)],'color','r');
  hold off; xlabel('X'); ylabel('Y'); zlabel('Z'); title('Problem schema in 3D');
  legend('Ball','Center (1,1,1)','Point x0 = [-3, 2.5, 3]',...,
      'Projection of x0 onto the Ball','Location','NorthEast');


  
  
 %% Excercise 3 - 2D DRAWINGS

  a = [7,8]; x0 = [3,3]; x1 = [11,6];
  x_interior = [10,10];
  
  x_star0(1) = max(a(1),x0(1));
  x_star0(2) = max(a(2),x0(2));
  
  x_star1(1) = max(a(1),x1(1));
  x_star1(2) = max(a(2),x1(2));  
  
  % Create and plot rectangle (where x>= a) and projections
  figure(); axis([0,12,0,12]);
  rectangle('Position',[a,12,12],'FaceColor','m');
  hold on; plot(x0(1),x0(2),'xk');
  plot(x_star0(1),x_star0(2),'*k');
  plot(x1(1), x1(2),'xb');
  plot(x_star1(1),x_star1(2),'*b');
  plot(x_interior(1), x_interior(2), 'xr');
  plot(x_interior(1), x_interior(2), '*r');
  line([x0(1), x_star0(1)], [x0(2), x_star0(2)], 'color', 'k');
  line([x1(1), x_star1(1)], [x1(2), x_star1(2)], 'color', 'b');
  hold off; xlabel('X'); ylabel('Y'); title('Problem schema in 2D');
  legend('Point x0 = [3,3]', 'Projection of x0','Point x1 = [11,6]','Projection of x1', ...,
      'Point x2 = [10,10]','Projection of x2','Location', 'NorthWest');



  
 %% Excercise 3 - 3D DRAWINGS
 
  a = [3,3,3];
  
  x0 = [0,3.5,1];
  x1 = [4,0,3.5];
  x2 = [4,4,1];
  
  for i = 1 : 3
      x_star0(i) = max(a(i),x0(i));
      x_star1(i) = max(a(i),x1(i));
      x_star2(i) = max(a(i),x2(i));  %#ok<SAGROW>
      
  end    

  % Create and plot rectangle (where x>= a) and projections  
  vert = [a;4 3 3;4 4 3;3 4 3;3 3 4;4 3 4;4 4 4;3 4 4];
  fac = [1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8];
  figure(); patch('Vertices',vert,'Faces',fac,'FaceVertexCData',hsv(8),'FaceColor','interp');
  view(3); axis vis3d; grid on; axis([0,4,0,4,0,4]);
  hold on; plot3(x0(1), x0(2), x0(3), 'xk');
  plot3(x_star0(1), x_star0(2), x_star0(3), '*k');
  plot3(x1(1), x1(2), x1(3), 'xr');
  plot3(x_star1(1), x_star1(2), x_star1(3), '*r');
  plot3(x2(1), x2(2), x2(3), 'xb');
  plot3(x_star2(1), x_star2(2), x_star2(3), '*b');  
  line([x0(1),x_star0(1)], [x0(2),x_star0(2)], [x0(3),x_star0(3)], 'color', 'k');
  line([x1(1),x_star1(1)], [x1(2),x_star1(2)], [x1(3),x_star1(3)], 'color', 'r');
  line([x2(1),x_star2(1)], [x2(2),x_star2(2)], [x2(3),x_star2(3)], 'color', 'b');
  hold off; xlabel('X'); ylabel('Y'); zlabel('Z'); title('Problem schema in 3D');
  legend('Rectangle', 'Point x0 = [0,3.5,1]', 'Projection of x0','Point x1 = [4,0,3.5]',...,
      'Projection of x1','Point x2 = [4,4,1]','Projection of x2');
  
      
  
 %% Excercise 4 - using the projected gradient descent method
  
%               The projected gradient descent method
%
%  x_{k+1} = P_{H}(x_k-1/L grad(f(x_k))), where L:=max(eig(Hessian(f))
%
  clear; n = 2; epsilon = 10^(-3);
    
  % Function f_0(x) = 1/2 ||x||^2
  f_0 = @(x) 0.5*(x'*x);
  
  % Gradient of f_0 is equal to x
  grad_f_0 = @(x) x;
  
  % L:=max(eig(Hessian(f)), where Hessian of f_0 is equal to 1 
  L = max(eig(ones(n,n)));
  
  % P_H(x) := projection of x onto the set H:={a^T x - b = 0}
  P_H = @(x,a,b) x - (a'*x-b)*a/(norm(a)^2); 
 
  % Construct random vectors a and b = a^T x
  a = unifrnd(0,1,n,1);
  x = unifrnd(0,1,n,1); b = a'*x;    
  
  % Initial point x_0 
  xk = 10*randn(n,1); k = 0;
  
  fprintf(' EXERCISE 4 \n \n');
    
  while (1)  
               
      xkk = P_H( xk-(1/L)*grad_f_0(xk),a,b );
          
      k = k + 1;
            
      fprintf('   Iter = %3d, f(xkk) = %4f, norm(grad) = %4f \n',k, f_0(xkk),...,
          ( norm(xk - xkk) / norm(xk) ) );
       
      if ( norm(xk - xkk) / norm(xk) ) < epsilon
        
          break
          
      end      
      
      xk = xkk;
      
  end
    
  x_star_grad = xkk;
    
  fprintf('   Optimal value = %4f \n',f_0(x_star_grad));
  fprintf('   Number of iterations =%3d \n\n',k);
    
  
  % Now let's solve this using the cvx to make sure it is correct
  cvx_begin quiet 
     variable x(n)
     minimize( f_0(x) )
     subject to
      a' * x == b; %#ok<EQEFF>
  cvx_end
    
 x_cvx = x; 
 fprintf(' Confirming the accuracy of our solution by solving with cvx \n');
 fprintf(' Optimal value = %4f \n \n \n',f_0(x_cvx));    
  
  
  
 %% Excercise 5 - using cvx 
    clear;
    p = 3; n = 2; K = 10; fprintf(' EXERCISE 5 \n \n You are using p = %d, n = %d and K = %d. \n \n',p,n,K);
    
    % Construct random vector q
    q = rand(n,1);
  
    % Construct positive definite matrix P (n x n)
    [U,S,V] = svd(unifrnd(0,1,n,n)); Lmin = 1; Lmax = K * Lmin;
    eigenvalues = [Lmin; Lmax; Lmin + (Lmax-Lmin) * unifrnd(0,1,n-2,1)];
    L = diag(eigenvalues); P = U*L*U';

    % Function f
    f = @(x) 0.5*x'*P*x + q'*x;
    
    % Construct random matrix A (p x n)
    A = unifrnd(0,1,p,n);
    
    % Construct random vector b as b = Ax
    x = unifrnd(0,1,n,1); b = A*x;    
  
    cvx_begin quiet
        variable x(n)
        minimize( f(x) )
        subject to
        A * x == b; %#ok<EQEFF>
    cvx_end
    
    x_star_cvx = x;
  
    fprintf(' <strong> (i)   </strong> Using the cvx \n');
    fprintf('         Optimal value = %4f \n \n',f(x_star_cvx));    

    
  %% Excercise 5 - using KKT conditions
  
  matrix1 = [-q ; b]; % dimension is (n+p) x 1
  matrix2 = [P A'; A zeros(p,p)]; % dimension is (n+p) x (n+p)

  temp = pinv(matrix2)*matrix1;
 
  x_star_KKT = temp(1:n);
  
  fprintf(' <strong> (ii)  </strong> Using the KKT conditions \n');
  fprintf('         Optimal value = %4f \n \n',f(x_star_KKT));
  
  
 %% Excercise 5 - using the projected gradient descent method
  
%               The projected gradient descent method
%
%  x_{k+1} = P_{S}(x_k-1/L grad(f(x_k))), where L:=max(eig(Hessian(f))
%
  
  epsilon = 10^(-3);
    
  % Initial point x_0 
  xk = 10*randn(n,1); 
  
  % Gradient of f_0 is equal to P*x + q
  grad_f = @(x,P,q) P*x + q;
  
  % Hessian of f_0 is equal to P 
  hessian_f = P;
  
  % L:=max(eig(Hessian(f))
  L = max(eig(hessian_f));
  
  % P_S(x) := projection of x onto the set S:={Ax-b=0}
  P_S = @(x,A,b) x - A'*(pinv(A*A')*(A*x-b));
   
  k = 0;
  fprintf(' <strong> (iii) </strong> Using the projected gradient descent method \n');
    
  while (1)  
               
      xkk = P_S( xk-(1/L)*grad_f(xk,P,q),A,b );
          
      k = k + 1;
            
      fprintf('         Iter = %3d, f(xkk) = %4f, norm(grad) = %4f \n',k, f(xkk),...,
          ( norm(xk - xkk) / norm(xk) ) );
       
      if ( norm(xk - xkk) / norm(xk) ) < epsilon
        
          break
          
      end      
      
      xk = xkk;
      
  end
    
  x_star_grad = xkk;
    
  fprintf('         Optimal value = %4f \n',f(x_star_grad));
  fprintf('         Number of iterations =%3d \n\n',k);
    
      
  
 
 