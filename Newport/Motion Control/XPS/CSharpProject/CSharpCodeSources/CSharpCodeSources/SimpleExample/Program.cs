using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleExample
{
    static class Program
    {
        /// <summary>
        /// Point d'entrée principal de l'application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            string strAssemblyPath = string.Empty;
            string logINIPath = string.Empty;
            string LogINIFullPath = string.Empty;
            string LOG_FILE_NAME = "Newport.XPS.Log.txt";

            strAssemblyPath = System.IO.Directory.GetParent(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)).FullName;
            logINIPath = "\\XPS\\";
            LogINIFullPath = strAssemblyPath + logINIPath + LOG_FILE_NAME;
            
            try
            {
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                Application.Run(new FirmwareVersionGet());
            }
            catch (Exception ex)
            {
                System.IO.File.WriteAllText(LogINIFullPath, ex.ToString());
                MessageBox.Show("An exception has occurred. For more information, refer to the file: " + LogINIFullPath);
            }
        }
    }
}
