% The ; denotes we are going back to a new row.
A = [1,2,3;4,5,6;7,8,9;10,11,12];

% Initializing a vector
v=[1;2;3];

% to get the dimensions of the matrix A where m=rows and n=columns
[m,n]=size(A);

% storing the dimension in other way
dim_A=size(A);

% to get the dimension of the vector v
dim_v=size(v);

% to index second row and 3rd column of the matrix A
A_23=A(2,3);

