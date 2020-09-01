using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using HDF.PInvoke;

using MathNet.Numerics.LinearAlgebra;


namespace controller
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        ModelPredictiveControl mpc = new ModelPredictiveControl();

        public MainWindow()
        {
            InitializeComponent();

            mpc.LoadParameters();




            //using (StreamWriter sw = new StreamWriter(@"C:\Users\alexa\Documents\Orgerlab Bitbucket\tracking_scope\controller\testmpc.txt"))
            //{

            Stopwatch sw = new Stopwatch();

            
                for (int i = 0; i < 1000; i++)
                {
                    sw.Restart();
                    mpc.UpdateModel();
                    sw.Stop();
                Debug.WriteLine(sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency);
                }
            //}








        }
    }
}
