clear all,close all,clc
num_agents = 1
load('D:\multi\env/Result/mat_height.txt')
load('D:\multi\env/Result/mat_exist.txt')

flag = 2;
Circle_3D_big;

    
color = str2mat('r','g','b','c','m','y','k','r','g','b','c','m','y','k','r','g','b','c','m','y','k','r','g','b','c','m','y','k')

for num = 1:num_agents
    
    load(['D:\multi\env/Result/path_agent',num2str(num-1),'.txt'])
    
    L=100;
    N=100;
    

    path_agent = eval(['path_agent',num2str(num-1)]);
    hold on;
    scatter3(path_agent(:,1),path_agent(:,2),100*ones(length(path_agent(:,1)),1),300,'r','*');
    scatter3(path_agent(1,3),path_agent(1,4),100*ones(length(path_agent(1,3)),1),150,'filled','r','o');
    %scatter3(path_agent1(:,1),path_agent1(:,2),100,'filled','m');


    %  for i=1:length(path_agent1(:,3))
    %      if atanh(path_agent1(i,17))*10<8
    %          scatter3(path_agent1(i,3),path_agent1(i,4),100*ones(length(path_agent1(i,3)),1),20,'filled','b');
    %      else
    %          scatter3(path_agent1(i,3),path_agent1(i,4),100*ones(length(path_agent1(i,3)),1),20,'filled','r');
    %      end
    %  end

    scatter3(path_agent(:,3),path_agent(:,4),100*ones(length(path_agent(:,3)),1),20,'filled',color(num));

    hold on;
    [m,n] = size(path_agent);
    for i=1:20:m
        orient_temp = atan(path_agent(i, 17)/path_agent(i, 18));
        if path_agent(i, 18)>0
            if path_agent(i,17)>0
                orient = orient_temp/2/pi
            else
                orient = (orient_temp + 2*pi)/2/pi
            end
        else
            orient = (orient_temp + pi)/2/pi
        end
        for j=1:9
            theta = mod(orient*2*pi-pi/2+(j-1)*pi/8, 2*pi);
            dis = path_agent(i, j+4)*L;
            x = dis*sin(theta);
            hold on;
            plot( dis*sin(theta)*(0:1/N:1) +path_agent(i, 3), dis*cos(theta)*(0:1/N:1)+path_agent(i,4),'b' );
        end

        %theta = mod(path_agent(i, 15)*2*pi-pi/2+(j-1)*pi/8, 2*pi);
        disx=path_agent(i,3)-path_agent(i,1);
        disy=path_agent(i,4)-path_agent(i,2);
        hold on;
        plot(-disx*(0:1/N:1)+path_agent(i,3), -disy*(0:1/N:1)+path_agent(i,4),'g')
    end

end










