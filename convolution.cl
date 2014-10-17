__kernel void convolution(__global float * A, __global float * B, __global float * C, int n, int m)
{
   int ind = get_global_id(0);
   if(ind < n * n){
       int i = ind / n;
       int j = ind % n;
       float res = 0;
       for(int k = -(m/2); k <= m/2; ++k){
           for(int l = -(m/2); l <= m/2; ++l){
                if((i+k >= 0) &&
                   (i+k < n) &&
                   (j+l >= 0) &&
                   (j+l < n)){
                    res += A[(j + l) + (i + k) * n] * B[(l + m/2) + (k + m/2) * m];
                }
           }
       }
       C[ind] = res;
   }
}