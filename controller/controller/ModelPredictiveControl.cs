using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


using HDF.PInvoke;


namespace controller
{
    public class ModelPredictiveControl
    {
        int T;

        Matrix<double> phiPast;
        Matrix<double> errorInversion;
        Vector<double> pastCommands;
        Vector<double> predictedFishTrajectory;


        Vector<double> completeFishTrajectory;


        public void LoadParameters(/*string file*/)
        {
            string file = @"C:\Users\alexa\Documents\Orgerlab Bitbucket\tracking_scope\mpc_params.mat";

            HDF5.ReadDataset(file, "T", out T);
            HDF5.ReadDataset(file, "N", out int N);

            double[,] arrPhiPast = new double[N-1,T];
            HDF5.ReadDataset(file, "phi_past", out arrPhiPast);
            phiPast = Matrix<double>.Build.DenseOfArray(arrPhiPast).Transpose();

            double[,] arrResponseConstant = new double[T, T];
            HDF5.ReadDataset(file, "response_constant", out arrResponseConstant);
            errorInversion = Matrix<double>.Build.DenseOfArray(arrResponseConstant).Transpose(); ;

            pastCommands = Vector<double>.Build.Dense(N - 1);

            predictedFishTrajectory = Vector<double>.Build.Dense(T);

            string file2 = @"C:\Users\alexa\Documents\Orgerlab Bitbucket\tracking_scope\mpc_params_traj.mat";
            double[] arrTrajectory = new double[41999];
            HDF5.ReadDataset(file2, "x_trajectory", out arrTrajectory);
            completeFishTrajectory = Vector<double>.Build.DenseOfArray(arrTrajectory);

        }





        int t = 0;
        void PredictFishTrajectory()
        {
            predictedFishTrajectory = completeFishTrajectory.SubVector(t++, T);
        }



        public double UpdateModel()
        {
            PredictFishTrajectory();

            var predictedStageTrajectory = phiPast * pastCommands;

            var predictedError = predictedFishTrajectory - predictedStageTrajectory;

            var futureCommands = errorInversion * predictedError;

            var nextCommand = futureCommands[0];

            for (int i = 0; i < pastCommands.Count -1; i++)
            {
                pastCommands[i] = pastCommands[i + 1];
            }
            pastCommands[pastCommands.Count-1] = nextCommand;

            return nextCommand;
        }
    }



    public static class HDF5
    {



        //   static void GetDims()
        //{
        //    var space = H5D.get_space(datasetID);
        //    var ndims = H5S.get_simple_extent_ndims(space);
        //    ulong[] dims = new ulong[ndims];
        //    ulong[] maxdims = new ulong[ndims];
        //    H5S.get_simple_extent_dims(space, dims, maxdims);
        //}

        public static void ReadDataset(string file, string dataset, out double[] data)
        {
            var fileID = H5F.open(file, H5F.ACC_RDONLY);
            var datasetID = H5D.open(fileID, $"/{dataset}");
            unsafe
            {
                fixed (double* ptr = data)
                {
                    var status = H5D.read(datasetID, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, (IntPtr)ptr);
                }
            }
            H5D.close(datasetID);
            H5F.close(fileID);
        }


        public static void ReadDataset(string file, string dataset, out double[,] data)
        {
            var fileID = H5F.open(file, H5F.ACC_RDONLY);
            var datasetID = H5D.open(fileID, $"/{dataset}");
            unsafe
            {
                fixed (double* ptr = data)
                {
                    var status = H5D.read(datasetID, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, (IntPtr)ptr);
                }
            }
            H5D.close(datasetID);
            H5F.close(fileID);
        }

        public static void ReadHDF5Dataset(string file, string dataset, out double data)
        {
            double[,] tmp = new double[1, 1];
            ReadDataset(file, dataset, out tmp);
            data = tmp[0, 0];
        }

        public static void ReadDataset(string file, string dataset, out int data)
        {
            ReadHDF5Dataset(file, dataset, out double tmp);
            data = (int)tmp;
        }
    }
}
