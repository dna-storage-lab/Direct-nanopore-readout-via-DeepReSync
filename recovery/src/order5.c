/*
Date: Wed, 14 Aug 96 16:25:18 JST
From: Marc Fossorier <marc>
Message-Id: <9608140725.AA21147@imailab.iis.u-tokyo.ac.jp>
To: robert
*/
/* This function "order5_dec" written by Marc Fossorier performs
   order-5 decoding as described by him in his paper. "Gg_t" is the
   two-dimensional integer array containing the generator matrix of
   the code. "R" is the length "N" double-valued received vector
   containing the received real numbers. "out_D" is the result of
   order-5 decoding. */
// ------------------------------------------------------------------------
// This program is complementary material for the book:
//
// R.H. Morelos-Zaragoza, The Art of Error Correcting Coding, Wiley, 2002.
//
// ISBN 0471 49581 6
//
// This and other programs are available at http://the-art-of-ecc.com
//
// You may use this program for academic and personal purposes only.
// If this program is used to perform simulations whose results are
// published in a journal or book, please refer to the book above.
//
// The use of this program in a commercial product requires explicit
// written permission from the author. The author is not responsible or
// liable for damage or loss that may be caused by the use of this program.
//
// Copyright (c) 2002. Robert H. Morelos-Zaragoza. All rights reserved.
// ------------------------------------------------------------------------



# include "order.h"
# include "math.h"

void order5(Gg_t,R,out_D)
    int Gg_t[K][N];
    double R[N],out_D[N];
{
    int a,b,c,d,f;
    int r,mul_G,e_c1,e_c2,e_c3,e_c4,e_c5,e_counter,counter;
    int count_comp,loss,gain;
    int c_dh0,c_dh1,c_dh2,c_dh3,c_dh4,c_dh5;
    double resource_init,resource_available,resource_right,cost_left;
    double resource_p[N-K],non_resource_p[N-K];
    double sum_tail_init,new_sum_tail;
    double resource_dh0,resource_dh1,resource_dh2,resource_dh3,resource_dh4,resource_dh5;
    double resource_dh0_candidate,resource_dh1_candidate,resource_dh2_candidate,resource_dh3_candidate,resource_dh4_candidate,resource_dh5_candidate,resource_candidate;
    int index_p[N-K];
    double cost_I[K],cost_row[K];
    int step,temp,iter,k,l,m,index;
    double cost,tot,Max,min,opti;
    CHANGE change;
    double abs_R[N],C[N][2];
    int Dd[N],DD[N],GG[K][N],i,j,Gg[K][N];
    int permutation_final[N],permutation_I[K],permutation_R[N];
    double zero[K],fabs();
    void peterson_I_0(),quick_sort_track_0();
    void switch_column_I_0();
    void switch_vector_0(),switch_matrix_0();
 
   for (i=0;i < K;i++)
	for (j=0;j < N;j++)
	    Gg[i][j] = Gg_t[i][j]; 
   /*for (i=0;(i<I_IN_iovec_len);i++)*/
   for (i=0;i < N;i++)
   {
       abs_R[i] = (double) fabs((double) R[i]);
       if (i < K)
       {  
           permutation_I[i] = i;
       }  
       permutation_R[i] = i;
   }

   quick_sort_track_0(abs_R,permutation_R,0,N-1);

/* Note that the permutations returned by qs are from small to big ! */

/* Move the K independent most informative symbols to the front */
/* Column operations => Changes the code into an equivalent code */
 
   switch_vector_0(R,permutation_R,N-1);
 
   switch_matrix_0(Gg,permutation_R,K-1,N-1);
 
/* Put into systematic code */
/* Row operations => Keeps the code */
 
   peterson_I_0(Gg,zero);
                           /* Obtain the K columns of I */
 
   quick_sort_track_0(zero,permutation_I,0,K-1);

/* Switch the rows to obtain the K I_rows in the right order
   => keeps the code */
 
   for (i=0; i<K; i++)
   {
       for (j=0; j<N; j++)
       { 
           GG[i][j] = Gg[permutation_I[i]][j];
       }
   }
 
 
/* Switch the column to obtain the K I_column in a consecutive order
   => changes the code */
 
   switch_column_I_0(GG,Gg,R,permutation_R,permutation_final);
 
/* Compute 2N costs and K most probable bits */
 
   resource_init = 0.0;                         
   k = 0;
   l = 0;
   c_dh0 = 0;
   c_dh1 = 0;
   c_dh2 = 0;
   c_dh3 = 0;
   c_dh4 = 0;
   c_dh5 = 0;
   count_comp = 0;
   counter = 0;
   e_counter = 0;

   for (i=0; i<N; i++)
   {
       C[i][0] = (R[i] - 1.0) * (R[i] - 1.0);
       C[i][1] = (R[i] + 1.0) * (R[i] + 1.0);
       if (i < K)
       { 
           if (C[i][0] < C[i][1])
           {
              Dd[i] = 0;
              DD[i] = 1;
           }
           else
           {
              Dd[i] = 1;
              DD[i] = 0;
           }
           cost_I[i] = C[i][Dd[i]] - C[i][DD[i]]; /* start with <0 as HD */
       }
       else
       {
          Dd[i] = 0;
          for (j=0; j<K; j++)
          {
             Dd[i] = Dd[i] + Dd[j] * Gg[j][i];
          }
          Dd[i] = Dd[i] % 2;
          DD[i] = (Dd[i]+1) % 2;
          if (C[i][Dd[i]] > C[i][DD[i]])
          {
             index_p[k] = i;
             resource_p[k] = C[i][Dd[i]] - C[i][DD[i]];
             resource_init = resource_init + resource_p[k];
             e_counter++;
             k++;
          }
          else
          {
             non_resource_p[l] = C[i][DD[i]] - C[i][Dd[i]];
             l++;          /* to be used in next loop too */
          }
       }
    }

    sum_tail_init = 0.0;
    resource_dh0 = 0.0;
    resource_dh1 = 0.0;
    resource_dh2 = 0.0;
    resource_dh3 = 0.0;
    resource_dh4 = 0.0;
    resource_dh5 = 0.0;

    for (i=0,j=l-1;i<D_H-e_counter;i++,j--)
    {
        if (i < D_H-e_counter-1)
        {
           resource_dh0 = resource_dh0 + non_resource_p[j];
           c_dh0++;
        }
        if (i < D_H-e_counter-2)
        {
           resource_dh1 = resource_dh1 + non_resource_p[j];
           c_dh1++;
        }
        if (i < D_H-e_counter-3)
        {
           resource_dh2 = resource_dh2 + non_resource_p[j];
           c_dh2++;
        }
        if (i < D_H-e_counter-4)
        {
           resource_dh3 = resource_dh3 + non_resource_p[j];
           c_dh3++;
        }
        if (i < D_H-e_counter-5)
        {
           resource_dh4 = resource_dh4 + non_resource_p[j];
        }
        if (i < D_H-e_counter-6)
        {
           resource_dh5 = resource_dh5 + non_resource_p[j];
        }
        sum_tail_init = sum_tail_init + non_resource_p[j];
    }    

      Max = 0.0;
      step = 0;
      resource_available = resource_init;
      change.n_change = 0;
      i=K-1;
      // while (((-cost_I[i]+resource_dh0) < resource_available) && (i>=0))
      while ((i >= 0) && ((-cost_I[i] + resource_dh0) < resource_available))
      {
 
          k = 0;
          resource_right = resource_available- resource_p[k];
          cost_left = -cost_I[i];
          loss = 0;
          gain = 0;
          e_c1 = e_counter;
          cost_row[i] = cost_I[i];
          resource_candidate = - cost_row[i];
          for (l=0,j=K; j<N; j++)
          {
             cost_row[i] = cost_row[i] + (C[j][Dd[j]] - C[j][DD[j]]) * ((double) Gg[i][j]);
             if ((Gg[i][j] == 1) && (C[j][Dd[j]] - C[j][DD[j]] > 0.0))
             {
                 non_resource_p[l] = C[j][Dd[j]] - C[j][DD[j]];
                 l++;        /* to be used later */
                 e_c1--;
                 loss++;
             }
             if ((Gg[i][j] == 1) && (C[j][Dd[j]] - C[j][DD[j]] <= 0.0))
             {
                 e_c1++;
                 gain++;
                 resource_candidate = resource_candidate - (C[j][Dd[j]] - C[j][DD[j]]);
                 /* new contribution */
             }
             if ((Gg[i][j] == 0) && (C[j][Dd[j]] - C[j][DD[j]] > 0.0))
             {
                 resource_candidate = resource_candidate + (C[j][Dd[j]] - C[j][DD[j]]);
                 /* unchanged contribution */
             }
             if ((Gg[i][j] == 0) && (C[j][Dd[j]] - C[j][DD[j]] <= 0.0))
             {
                 non_resource_p[l] = C[j][DD[j]] - C[j][Dd[j]];
                 l++;        /* to be used later */
             }
          }
          if (cost_row[i] > Max)
          {
              Max = cost_row[i];
              change.n_change = 1;
              change.position[0] = i;
              resource_available = resource_init - Max;
 
/* Recompute resource_dhi for new resource_candidate */
 
               resource_dh0 = 0.0;
               resource_dh1 = 0.0;
               resource_dh2 = 0.0;
               resource_dh3 = 0.0;
               resource_dh4 = 0.0;
               resource_dh5 = 0.0;
                 
               // for (a=0,b=l-1;a<D_H-e_c1-1;a++,b--)  /* -1 as undo Order 1 */
               // { 
               //     if (a < D_H-e_c1-1-1)
               //     {
               //        resource_dh0 = resource_dh0 + non_resource_p[b];
               //     }
               //     if (a < D_H-e_c1-1-2)
               //     {
               //        resource_dh1 = resource_dh1 + non_resource_p[b];
               //     }
               //     if (a < D_H-e_c1-1-3)
               //     {
               //        resource_dh2 = resource_dh2 + non_resource_p[b];
               //     }
               //     if (a < D_H-e_c1-1-4)
               //     {
               //        resource_dh3 = resource_dh3 + non_resource_p[b];
               //     }
               //     if (a < D_H-e_c1-1-5)
               //     {
               //        resource_dh4 = resource_dh4 + non_resource_p[b];
               //     }
               //     if (a < D_H-e_c1-1-6)
               //     {
               //        resource_dh5 = resource_dh5 + non_resource_p[b];
               //     }
               //     sum_tail_init = sum_tail_init + non_resource_p[b];
               // }
               
               int limit = D_H - e_c1 - 1;   /* 原来的循环上界 */
               if (limit < 0) limit = 0;
               
               for (a = 0, b = l - 1; (a < limit) && (b >= 0); a++, b--)
               {
                   if (a < limit - 1) resource_dh0 += non_resource_p[b];
                   if (a < limit - 2) resource_dh1 += non_resource_p[b];
                   if (a < limit - 3) resource_dh2 += non_resource_p[b];
                   if (a < limit - 4) resource_dh3 += non_resource_p[b];
                   if (a < limit - 5) resource_dh4 += non_resource_p[b];
                   if (a < limit - 6) resource_dh5 += non_resource_p[b];
               
                   sum_tail_init += non_resource_p[b];
               }
               
           }
           i--;
      }
      step = 1;

      for (i=K-1;i>0;i--)
      {
         for (j=i-1;j>=0;j--)
         {
            if ((-cost_I[i]-cost_I[j]+resource_dh1) < resource_available)
            {

          k = 0;  
          resource_right = resource_available- resource_p[k];
          cost_left = -cost_I[i]-cost_I[j];
          loss = 0;
          gain = 0;
          e_c2 = e_counter;
          cost = cost_I[i]+cost_I[j];   /* negative value */
          resource_candidate = - cost;
          for (l=0,k=K; k<N; k++)
          {
             if (Gg[i][k] != Gg[j][k])
             {
                cost = cost + (C[k][Dd[k]] - C[k][DD[k]]);
                if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
                {
                    non_resource_p[l] = C[k][Dd[k]] - C[k][DD[k]];
                    l++;        /* to be used later */
                    e_c2--;
                    loss++;
                }
                else
                {
                    e_c2++;
                    gain++;
                    resource_candidate = resource_candidate - (C[k][Dd[k]] - C[k][DD[k]]);
                    /* new contribution */
                }
             }
             else
             {
                if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
                {
                    resource_candidate = resource_candidate + (C[k][Dd[k]] - C[k][DD[k]]);
                    /* unchanged contribution */
                }
                else
                {
                    non_resource_p[l] = C[k][DD[k]] - C[k][Dd[k]];
                    l++;        /* to be used later */
                }
             }
          }
          if (cost > Max)
          {
              Max = cost;          /* do not reset Max */
              change.n_change = 2;
              change.position[0] = i;
              change.position[1] = j;
              resource_available = resource_init - Max;
/* Recompute resource_dhi for new resource_candidate */
 
                  resource_dh0 = 0.0;
                  resource_dh1 = 0.0;
                  resource_dh2 = 0.0;
                  resource_dh3 = 0.0;
                  resource_dh4 = 0.0;
                  resource_dh5 = 0.0;

                  for (a=0,b=l-1;a<D_H-e_c2-2;a++,b--)  /* -2 as undo Order 2 */                  {
                      if (a < D_H-e_c2-2-1)
                      {
                         resource_dh0 = resource_dh0 + non_resource_p[b];
                      }
                      if (a < D_H-e_c2-2-2)
                      {
                         resource_dh1 = resource_dh1 + non_resource_p[b];
                      }
                      if (a < D_H-e_c2-2-3)
                      {
                         resource_dh2 = resource_dh2 + non_resource_p[b];
                      }
                      if (a < D_H-e_c2-2-4)
                      {
                         resource_dh3 = resource_dh3 + non_resource_p[b];
                      }
                      if (a < D_H-e_c2-2-5)
                      {
                         resource_dh4 = resource_dh4 + non_resource_p[b];
                      }
                      if (a < D_H-e_c2-2-6)
                      {
                         resource_dh5 = resource_dh5 + non_resource_p[b];
                      }
                      sum_tail_init = sum_tail_init + non_resource_p[b];
                  }
              }
             }
             else
             {
                j = -1;
             }
       }            /* end j */
      }             /* end i */
      step = 2;

      for (i=K-1;i>1;i--)             
      {
        for (j=i-1;j>0;j--)
        {
          for (l=j-1;l>=0;l--)
          {
            if ((-cost_I[i]-cost_I[j]-cost_I[l]+resource_dh2) < resource_available)
            {
 
      k = 0;
      resource_right = resource_available- resource_p[k];
      cost_left = -cost_I[i]-cost_I[j]-cost_I[l];
      loss = 0; 
      gain = 0;
      e_c3 = e_counter;
      cost = cost_I[i]+cost_I[j]+cost_I[l];   /* negative value */
      resource_candidate = - cost;
      for (c=0,k=K; k<N; k++)
      {
         mul_G = (Gg[i][k] + Gg[j][k] + Gg[l][k]) % 2;
         if (mul_G == 1)
         {
            cost = cost + (C[k][Dd[k]] - C[k][DD[k]]);
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0) 
            { 
                non_resource_p[c] = C[k][Dd[k]] - C[k][DD[k]];
                c++;        /* to be used later */
                e_c3--; 
            loss++;
            } 
            else
            { 
                e_c3++; 
                gain++;
                resource_candidate = resource_candidate - (C[k][Dd[k]] - C[k][DD[k]]);                        
                /* new contribution */
            } 
         }
         else 
         {  
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
            {   
                resource_candidate = resource_candidate + (C[k][Dd[k]] - C[k][DD[k]]);      
                /* unchanged contribution */
            }   
            else
            {    
                non_resource_p[c] = C[k][DD[k]] - C[k][Dd[k]];
                c++;        /* to be used later */
            }   
         }   
      }  

      if (cost > Max)
      {
          Max = cost;          /* do not reset Max */
          change.n_change = 3;
          change.position[0] = i;
          change.position[1] = j;
          change.position[2] = l;
          resource_available = resource_init - Max;

/* Recompute resource_dhi for new resource_candidate */ 
 
              resource_dh0 = 0.0; 
              resource_dh1 = 0.0; 
              resource_dh2 = 0.0; 
              resource_dh3 = 0.0;
              resource_dh4 = 0.0;
              resource_dh5 = 0.0;
              
              for (a=0,b=c-1;a<D_H-e_c3-3;a++,b--)  /* -3 as undo Order 3 */                  {                     
                  if (a < D_H-e_c3-3-1) 
                  {    
                     resource_dh0 = resource_dh0 + non_resource_p[b]; 
                  } 
                  if (a < D_H-e_c3-3-2) 
                  { 
                     resource_dh1 = resource_dh1 + non_resource_p[b]; 
                  } 
                  if (a < D_H-e_c3-3-3) 
                  { 
                     resource_dh2 = resource_dh2 + non_resource_p[b]; 
                  } 
                  if (a < D_H-e_c3-3-4) 
                  { 
                     resource_dh3 = resource_dh3 + non_resource_p[b]; 
                  } 
                  if (a < D_H-e_c3-3-5) 
                  { 
                     resource_dh4 = resource_dh4 + non_resource_p[b]; 
                  } 
                  if (a < D_H-e_c3-3-6) 
                  { 
                     resource_dh5 = resource_dh5 + non_resource_p[b]; 
                  } 
                  sum_tail_init = sum_tail_init + non_resource_p[b]; 
              } 
          } 
             }
             else
             {
                l = -1;
             }
         }          /* end l */
       }            /* end j */
      }             /* end i */
      step = 3;

      for (i=K-1;i>2;i--)
      {
        for (j=i-1;j>1;j--)
        {
          for (l=j-1;l>0;l--)
          {
            for (m=l-1;m>=0;m--)
            {
              if ((-cost_I[i]-cost_I[j]-cost_I[l]-cost_I[m]+resource_dh3) < resource_available)
              {
               
      k = 0;
      resource_right = resource_available- resource_p[k];
      cost_left = -cost_I[i]-cost_I[j]-cost_I[l]-cost_I[m];
      loss = 0;
      gain = 0;
      e_c4 = e_counter;
      cost = cost_I[i]+cost_I[j]+cost_I[l]+cost_I[m];   /* negative value */
      resource_candidate = - cost;
      for (c=0,k=K; k<N; k++)
      {
         mul_G = (Gg[i][k] + Gg[j][k] + Gg[l][k] + Gg[m][k]) % 2;
         if (mul_G == 1)
         {
            cost = cost + (C[k][Dd[k]] - C[k][DD[k]]);
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
            {
                non_resource_p[c] = C[k][Dd[k]] - C[k][DD[k]];
                c++;        /* to be used later */
                e_c4--;
                loss++;
            }
            else
            {
                e_c4++;
                gain++;
                resource_candidate = resource_candidate - (C[k][Dd[k]] - C[k][DD[k]]);       
                /* new contribution */
            }
         }  
         else
         {  
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
            {
                resource_candidate = resource_candidate + (C[k][Dd[k]] - C[k][DD[k]]);       
                /* unchanged contribution */
            }
            else
            {
                non_resource_p[c] = C[k][DD[k]] - C[k][Dd[k]];
                c++;        /* to be used later */
            }
         }   
      }  

      if (cost > Max)
      {
          Max = cost;          /* do not reset Max */
          change.n_change = 4;
          change.position[0] = i;
          change.position[1] = j;
          change.position[2] = l;
          change.position[3] = m;
          resource_available = resource_init - Max;

/* Recompute resource_dhi for new resource_candidate */

        resource_dh0 = 0.0;                   
        resource_dh1 = 0.0;
        resource_dh2 = 0.0;
        resource_dh3 = 0.0;
        resource_dh4 = 0.0;
        resource_dh5 = 0.0;

        for (a=0,b=c-1;a<D_H-e_c4-4;a++,b--)  /* -4 as undo Order 4 */          
        {
            if (a < D_H-e_c4-4-1)
            {
               resource_dh0 = resource_dh0 + non_resource_p[b];
            }
            if (a < D_H-e_c4-4-2)
            {
               resource_dh1 = resource_dh1 + non_resource_p[b];
            }
            if (a < D_H-e_c4-4-3)
            {
               resource_dh2 = resource_dh2 + non_resource_p[b];
            }
            if (a < D_H-e_c4-4-4)
            {
               resource_dh3 = resource_dh3 + non_resource_p[b];
            }
            if (a < D_H-e_c4-4-5)
            {
               resource_dh4 = resource_dh4 + non_resource_p[b];
            }
            if (a < D_H-e_c4-4-6)
            {
               resource_dh5 = resource_dh5 + non_resource_p[b];
            }
            sum_tail_init = sum_tail_init + non_resource_p[b];
        }    
      }
             }
             else
             {
                m = -1;
             }
           }        /* end m */
         }          /* end l */
       }            /* end j */
      }             /* end i */
      step = 4;

      for (i=K-1;i>3;i--)
      {
        for (j=i-1;j>2;j--)
        {
          for (l=j-1;l>1;l--)
          {
            for (m=l-1;m>0;m--)
            {
              for (f=m-1;f>=0;f--)
              {
                if ((-cost_I[i]-cost_I[j]-cost_I[l]-cost_I[m]-cost_I[f]+resource_dh4) < resource_available)
              {

      k = 0;
      resource_right = resource_available- resource_p[k];
      cost_left = -cost_I[i]-cost_I[j]-cost_I[l]-cost_I[m]-cost_I[f];
      loss = 0;
      gain = 0;
      e_c5 = e_counter;
      cost = cost_I[i]+cost_I[j]+cost_I[l]+cost_I[m]+cost_I[f];
                                                       /* negative value */
      resource_candidate = - cost;
      for (c=0,k=K; k<N; k++)
      {
         mul_G = (Gg[i][k] + Gg[j][k] + Gg[l][k] + Gg[m][k] + Gg[f][k]) % 2;
         if (mul_G == 1)
         {
            cost = cost + (C[k][Dd[k]] - C[k][DD[k]]);
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
            {
                non_resource_p[c] = C[k][Dd[k]] - C[k][DD[k]];
                c++;        /* to be used later */
                e_c5--;
                loss++;
            }
            else
            {
                e_c5++;
                gain++;
                resource_candidate = resource_candidate - (C[k][Dd[k]] - C[k][DD
[k]]);
                /* new contribution */
            }
         }
         else
         {
            if (C[k][Dd[k]] - C[k][DD[k]] > 0.0)
            {
                resource_candidate = resource_candidate + (C[k][Dd[k]] - C[k][DD
[k]]);
                /* unchanged contribution */
            }
            else
            {
                non_resource_p[c] = C[k][DD[k]] - C[k][Dd[k]];
                c++;        /* to be used later */
            }
         }
      }
      if (cost > Max)
      {
          Max = cost;          /* do not reset Max */
          change.n_change = 5;
          change.position[0] = i;
          change.position[1] = j;
          change.position[2] = l;
          change.position[3] = m;
          change.position[4] = f;
          resource_available = resource_init - Max;

/* Recompute resource_dhi for new resource_candidate */

        resource_dh0 = 0.0;
        resource_dh1 = 0.0;
        resource_dh2 = 0.0;
        resource_dh3 = 0.0;
        resource_dh4 = 0.0;
        resource_dh5 = 0.0;

        for (a=0,b=c-1;a<D_H-e_c5-5;a++,b--)  /* -5 as undo Order 5 */
        {
            if (a < D_H-e_c5-5-1)
            {
               resource_dh0 = resource_dh0 + non_resource_p[b];
            }
            if (a < D_H-e_c5-5-2)
            {
               resource_dh1 = resource_dh1 + non_resource_p[b];
            }
            if (a < D_H-e_c5-5-3)
            {
               resource_dh2 = resource_dh2 + non_resource_p[b];
            }
            if (a < D_H-e_c5-5-4)
            {
               resource_dh3 = resource_dh3 + non_resource_p[b];
            }
            if (a < D_H-e_c5-5-5)
            {
               resource_dh4 = resource_dh4 + non_resource_p[b];
            }
            if (a < D_H-e_c5-5-6)
            {
               resource_dh5 = resource_dh5 + non_resource_p[b];
            }
            sum_tail_init = sum_tail_init + non_resource_p[b];
        }
      }
               }
               else
               {
                  f = -1;
               }
             }      /* end f */
           }        /* end m */
         }          /* end l */
       }            /* end j */
      }             /* end i */
      step = 5;

/* Re permute to obtain the original sequence (use only permute_R as
   permute_I worked on rows). Use DD[] as DD[] no longer useful */

   for (i=0;i<change.n_change;i++)                                 
   {
       Dd[change.position[i]] = DD[change.position[i]];
   }

   for (i=K; i<N; i++)
   {
       Dd[i] = 0;
       for (j=0; j<K; j++)
       { 
          Dd[i] = Dd[i] + Dd[j] * Gg[j][i];
       } 
       Dd[i] = Dd[i] % 2;
   }    

   for (i=0; i<N; i++)
   {
       DD[permutation_final[i]] = Dd[i];
   }
 
   for (i=0; i < N; i++)   /* Using a systematic code */
   {
      if (DD[i] == 0)
         out_D[i] = - 1.0;
      else
         out_D[i] = 1.0;
   }

}


void switch_column_I_0(GG,Gg,R,permutation_R,permutation_final)
  /* copy GG into Gg */   
   int GG[K][N],Gg[K][N];
   double R[];
   int permutation_R[],permutation_final[];
{
    int i,j,index,count,start;
    int record[N],record_I[N];
    double temp[N];
 
    index = 0;
    count = 0;
 
    for (i=0; i < K+index; i++)   /* record the column positions to switch */
    {
         if (GG[i-index][i] == 0)
         {
              record[index] = i;
              index++;
         }
         else
         {
              record_I[count] = i;
              count ++;
         }
    }
    start = K + index - 1;
    i = N-1;
    while (i > start)            /* Unchanged part */
    {
        for(j=0; j<K; j++)
        {
            Gg[j][i] = GG[j][i];
            permutation_final[i] = permutation_R[N-1-i];
        }
        i--;
    }    
    while (index > 0)            /* Copy from R to L dependent columns found
                                    in first positions */
    {
        temp[i] = R[i];
        for(j=0; j<K; j++)
        {
            Gg[j][i] = GG[j][record[index-1]];
            permutation_final[i] = permutation_R[N-1-record[index-1]];
        }
        if (i < record[index-1])
        {
            R[i] = temp[record[index-1]];
        }
        else
        {
            R[i] = R[record[index-1]];
        }
        i--;
        index--;
    }
    while (count > 0)    /* Copy I_columns */
    {
        temp[i] = R[i];
        for(j=0; j<K; j++)
        {
            Gg[j][i] = GG[j][record_I[count-1]];
            permutation_final[i] = permutation_R[N-1-record_I[count-1]];
        }
 
        if (i < record_I[count-1])
        {
            R[i] = temp[record_I[count-1]];
        }
        else
        {
            R[i] = R[record_I[count-1]];
        } 
        i--;
        count--;
    }
}
 
void switch_vector_0(R,permutation,length)
   double R[];
   int permutation[];
   int length;
{
   int i;
   double temp[N];
 
   for (i=0;i<=length;i++)
   {
      temp[i] = R[i];
      if (permutation[length-i] > i)   /* data not overwritten */
      {
          R[i] = R[permutation[length-i]];
      }
      else
      {
          R[i] = temp[permutation[length-i]];
                                      /* data overwritten but stored in temp */
      }
   }
}
 
void switch_matrix_0(Gg,permutation,k,n)
   int Gg[K][N];
   int permutation[];
   int k,n;
{
   int i,j;
   double temp[K][N];
 
   for (j=0;j<=n;j++)
   {
      for (i=0; i<=k; i++)
      {
         temp[i][j] = Gg[i][j];
         if (permutation[n-j] > j)   /* column not overwritten */
         {
             Gg[i][j] = Gg[i][permutation[n-j]];
         }
         else
         {
             Gg[i][j] = temp[i][permutation[n-j]];
                                      /* data overwritten but stored in temp */
         }
      }
   }
}
 
void peterson_I_0(Gg,zero)
   int Gg[K][N];
   double zero[N];
{
   int i,j,l,m;
 
   for (i=0; i<K; i++)
   {
      j = 0;
      zero[i] = 0.0;
      while (Gg[i][j] == 0)
      {
         j++;
         zero[i] = zero[i] + 1.0;
      }
      for (l=0; l<K ; l++)
      {
          if ((l != i) && (Gg[l][j] == 1))
          {
               for (m=0; m<N; m++)
               {
                  Gg[l][m] = (Gg[l][m] + Gg [i][m])%2;
               }
          }
      }
   }
}
 

// /* Recursive quick_sort */
// void quick_sort_track_0(vals,permutation,left,right) /* Tracks the permutations */        
//         double vals[];
//         int permutation[];
//         int left,right;
// {
//     /* ascending = left wall, descending = right wall */

//     int ascending = left-1, descending = right;
//     double ref_val,temp;
//     int temp2;

//     /* select the comparison value */

//     if (right > left)                 
//     {
//         ref_val = vals [right];

//         for(;;)                 
//         {

//        /* while element smaller than reference value,
//           move left wall upwards
//        */

//              while (vals [++ascending] < ref_val);

//        /* while element larger than reference value,
//           move right wall downwards
//        */

//              while (vals [--descending] > ref_val);

//              if (ascending >= descending) break;    
 
//        /* if the walls have not passed each other,
//           exchange values
//           (if not access to function need to save index)
//        */

//             temp=vals[ascending];
//             temp2=permutation[ascending];
//             vals[ascending]=vals[descending];
//             vals[descending]=temp;
//             permutation[ascending]=permutation[descending];
//             permutation[descending]=temp2;

//          }                                 

           
//             temp=vals[ascending];
//             temp2=permutation[ascending];
//             vals[ascending]=vals[right];
//             vals[right]=temp;
//             permutation[ascending]=permutation[right];
//             permutation[right]=temp2;
 

//     /* if descending wall has not yet reached left partition,
//        call quick-sort again with this smaller array
//     */
 
//        quick_sort_track_0( vals, permutation, left, ascending-1);

//     /* if ascending wall has not yet reached right partition,   
//        call quick-sort again with this smaller array
//     */
 
//        quick_sort_track_0( vals, permutation, ascending+1, right);
//    }     

// }


void quick_sort_track_0(double vals[], int permutation[], int left, int right)
{
    int i = left;
    int j = right;
    double pivot = vals[(left + right) / 2];

    while (i <= j) {
        while (vals[i] < pivot) i++;
        while (vals[j] > pivot) j--;

        if (i <= j) {
            double tv = vals[i];
            vals[i] = vals[j];
            vals[j] = tv;

            int tp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tp;

            i++;
            j--;
        }
    }

    if (left < j)  quick_sort_track_0(vals, permutation, left, j);
    if (i < right) quick_sort_track_0(vals, permutation, i, right);
}
