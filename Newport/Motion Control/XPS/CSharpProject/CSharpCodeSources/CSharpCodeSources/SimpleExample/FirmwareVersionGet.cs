using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CommandInterfaceXPS; // Newport Assembly .Net access

namespace SimpleExample
{
    public partial class FirmwareVersionGet : Form
    {
        // Create communication interface to get firmware version
        CommandInterfaceXPS.XPS myController = new CommandInterfaceXPS.XPS(); 
        
        public FirmwareVersionGet()
        {
            InitializeComponent();
            errorMsgLabel.Text = "";
            firmwareVersionLabel.Text = "";
        }

        private void connectButtonButton_Click(object sender, EventArgs e)
        {
            // Open the communication interface and check if it succeded
            if (myController.OpenInstrument(IPAddressTextBox.Text, 5001, 100) != CommandInterfaceXPS.XPS.SUCCESS) // If communication interface can't be opened
            {
                // Display error message
                MessageBox.Show("Error while opening instrument.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                string errstring = string.Empty;
                string firmwareVersion = string.Empty;
                int answer = 0;

                // call FirmwareVersionGet() API
                answer = myController.FirmwareVersionGet(out firmwareVersion, out errstring);
                switch (answer)
                {
                    case CommandInterfaceXPS.XPS.SUCCESS:
                        firmwareVersionLabel.Text = firmwareVersion;
                        errorMsgLabel.Text = "Success"; 
                        break;
                    case CommandInterfaceXPS.XPS.FAILURE:
                        firmwareVersionLabel.Text = "";
                        errorMsgLabel.Text = "Failure"; 
                        break;
                    case CommandInterfaceXPS.XPS.TIMEOUT:
                        firmwareVersionLabel.Text = "";
                        errorMsgLabel.Text = "Time Out"; 
                        break;
                }
               
                // Close the instrument
                myController.CloseInstrument();
            }
        }
    }
}
