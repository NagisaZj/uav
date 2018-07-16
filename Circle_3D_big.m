N=30;
[Len,Win]=size(mat_height);
for i=0:Len-16
    for j=0:Win-16
        if (mat_exist(i+9,j+9)>0)
            height=mat_height(i+9,j+9);
            A=linspace(0,2*pi,100);
            Z=linspace(0,height,N);
            [A,Z]=meshgrid(A,Z);
            LX=60*cos(A)+200*i+100;
            LY=60*sin(A)+200*j+100;
            mesh(LX,LY,Z);hold on;

            x=linspace(-60,60,100);
            y=linspace(-60,60,100);
            [X,Y]=meshgrid(x,y);
            X(X.^2+Y.^2>3600)=NaN;
            Y(X.^2+Y.^2>3600)=NaN;
            mesh(X+200*i+100,Y+200*j+100,X*0);
            mesh(X+200*i+100,Y+200*j+100,X*0+height);
        end
    end
end