figure;
x = 0:.1:10;
y = abs(3-x) + abs(5-x) + abs(6-x);
plot(x,y,'-r','LineWidth',3);
xlabel('w')
ylabel('y')
title('Plot of f(w)')
