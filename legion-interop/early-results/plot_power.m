function [] = plot_power()
    M = load('infile3.txt');
    p1 = M(1:2:end,:);  % package-1
    p2 = M(2:2:end,:);  % package-0
    a1 = p1;
    a2 = p2;
    size1 = size(a1,1);
    size2 = size(a2,1);
    for i=size1:-1:2
        a1(i,2) = 1000000*( a1(i,2)-a1(i-1,2) ) / (a1(i,1)-a1(i-1,1));
    end
    for i=size2:-1:2
        a2(i,2) = 1000000*( a2(i,2)-a2(i-1,2) ) / (a2(i,1)-a2(i-1,1));
    end
    for i=size1:-1:1
        a1(i,1) = a1(i,1) - a1(1,1);
    end
    for i=size2:-1:1
        a2(i,1) = a2(i,1) - a2(1,1);
    end
    figure();
    plot(a1(2:end,1), a1(2:end,2)),xlabel('time [us]'),ylabel('power [W]'),title('Package Level Power Across Timesteps (256 Tasks)');
    hold on;
    plot(a2(2:end,1), a2(2:end,2));
    
    ystart = 15*[1 1 1 1 1 1 1 1 1 1];
    yend = 85*[1 1 1 1 1 1 1 1 1 1 1];
    % g1
    %xstep = [1521527557478041 1521527561267343 1521527564638005 1521527567956941 1521527571205190 1521527574429756]-p1(1,1);
    % g2
    %xstep = [1521529313434463 1521529319321023 1521529324186374 1521529329055239 1521529333839611 1521529338716101]-p1(1,1);
    % ignore, this was for index tasks. looks just like next one.
    %xstep = [1521572316429958 1521572322264759 1521572328049806 1521572333841542 1521572339724145 1521572345502905 1521572351277012 1521572357067426 1521572362853592 1521572368631038] - p1(1,1);
    % g3
    xstep = [1521781221861221 1521781224662742 1521781226944470 1521781229218832 1521781231506043 1521781233802650 1521781236088959 1521781238380745 1521781240673434 1521781242972711] - p1(1,1);
    
    for idx = 1:numel(xstep)
        plot([xstep(idx) xstep(idx)], [ystart(idx) yend(idx)], 'g'),legend('Location','southeast','package-1','package-0','timestep');
    end
    hold off;
end