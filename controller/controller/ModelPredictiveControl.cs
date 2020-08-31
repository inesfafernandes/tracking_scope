using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;



namespace controller
{
    public class ModelPredictiveControl
    {
        const int T = 250;
        const int sizeH = 500;

        Matrix<double> phi = Matrix<double>.Build.Dense(T, T + sizeH);



        void CreatePhi()
        {
            var m = Matrix<double>.Build.Dense(6, 4, (i, j) => 10 * i + j);

        }

        public void UpdateModel()
        {

            /*
             
        
        predicted_fish_trajectory=fish_trajectory(t:t+T-1);

        predicted_stage_trajectory=phi_a*up;

        predicted_error=predicted_fish_trajectory-predicted_stage_trajectory;

        uf=const_up*(predicted_error);%computing u future x axis
        
        up=cat(1,up,uf(1));% updating u past by adding the first element of uf as the last element of u past

        up(1)=[];% and discarding the first value of u past

        u=cat(1,u,uf(1));%vector that contains all commands that were sent
 
     
             
             
             */
        }




    }
}
