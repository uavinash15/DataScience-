A=[1,2,0;0,5,6;7,0,9];
% transposing A
A_transpose=A';

%inversing A
A_inverse=inv(A);

% multiplying Ainverse with A
A_invA=inv(A)*A;
invA_A=A*inv(A);
I1=A_inverse*A;
I2=A*A_inverse;