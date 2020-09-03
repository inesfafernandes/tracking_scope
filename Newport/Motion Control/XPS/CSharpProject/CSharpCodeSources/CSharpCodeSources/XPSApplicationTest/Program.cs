using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace XPSApplicationTest
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
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
                Application.Run(new FormApplicationXPS());
            }
            catch (Exception ex)
            {
                System.IO.File.WriteAllText(LogINIFullPath, ex.ToString());
                MessageBox.Show("An exception has occurred. For more information, refer to the file: " + LogINIFullPath);
            }
        }
    }
}
