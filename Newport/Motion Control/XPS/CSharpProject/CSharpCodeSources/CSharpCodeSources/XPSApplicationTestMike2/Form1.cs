using System;
using System.Collections.Generic;
using System.Globalization;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CommandInterfaceXPS; // Newport Assembly .Net access

namespace XPSApplicationTest
{
    public partial class FormApplicationXPS : Form
    {
        const int DEFAULT_TIMEOUT = 10000;
        const int POLLING_INTERVALLE_MS = 1; // Milliseconds
        const int NB_POSITIONERS = 1;

        CommandInterfaceXPS.XPS m_xpsInterface = null;           // Socket #1 (order)
        CommandInterfaceXPS.XPS m_xpsInterfaceForPolling = null; // Socket #2 (polling)

        string m_IPAddress;
        int      m_IPPort;
        bool     m_CommunicationOK;
        string   m_GroupName;
        string   m_PositionerName;
        string gpioname;
        bool     m_IsPositioner;
        double[] m_TargetPosition = new double[NB_POSITIONERS];
        double[] m_CurrentPosition = new double[NB_POSITIONERS];
        int      m_CurrentGroupStatus;
        string   m_CurrentGroupStatusDescription;
        string   m_XPSControllerVersion;
        string   m_errorDescription;

        int            m_PollingInterval;
        bool           m_pollingFlag;
        private Thread m_PollingThread;

        public delegate void ChangedCurrentPositionHandler(double[] currentPositions);
        private event ChangedCurrentPositionHandler m_CurrentPositionChanged;
        public event ChangedCurrentPositionHandler PositionChanged
        {
            add { m_CurrentPositionChanged += value; }
            remove { m_CurrentPositionChanged -= value; }
        }
        public delegate void ChangedCurrentGroupStateHandler(int currentGroupStatus, string description);
        private event ChangedCurrentGroupStateHandler m_CurrentGroupStateChanged;
        public event ChangedCurrentGroupStateHandler GroupStatusChanged
        {
            add { m_CurrentGroupStateChanged += value; }
            remove { m_CurrentGroupStateChanged -= value; }
        }
        public delegate void ChangedLabelErrorMessageHandler(string currentErrorMessage);
        private event ChangedLabelErrorMessageHandler m_ErrorMessageChanged;
        public event ChangedLabelErrorMessageHandler ErrorMessageChanged
        {
            add { m_ErrorMessageChanged += value; }
            remove { m_ErrorMessageChanged -= value; }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public FormApplicationXPS()
        {
            InitializeComponent();

            // Initialization
            label_MessageCommunication.ForeColor = Color.Red;
            label_MessageCommunication.Text = string.Format("Disconnected from XPS");
            m_IsPositioner = false;
            m_CommunicationOK = false;
             gpioname = string.Format("GPIO4.ADC1");
            m_pollingFlag = false;
            m_PollingInterval = POLLING_INTERVALLE_MS; // milliseconds
            m_CurrentGroupStatus = 0;
            for (int i = 0; i < NB_POSITIONERS; i++)
                m_TargetPosition[i] = 0;

            // Events
            if (this != null)
            {
                this.PositionChanged += new ChangedCurrentPositionHandler(CurrentPositionHandlerChanged);
                this.GroupStatusChanged += new ChangedCurrentGroupStateHandler(CurrentGroupStateHandlerChanged);
                this.ErrorMessageChanged += new ChangedLabelErrorMessageHandler(ErrorMessageHandlerChanged);
            }
        }
        
        private void CurrentPositionHandlerChanged(double[] currentValues)
        {
            string strPosition = currentValues[0].ToString("F2", CultureInfo.CurrentCulture.NumberFormat);
            
            textBoxPosition.BeginInvoke(
                   new Action(() =>
                   {
                       textBoxPosition.Text = strPosition;
                   }
                ));
        }

        private void CurrentGroupStateHandlerChanged(int currentGroupStatus, string strGroupStatusDescription)
        {
            try
            {
                string strStatus = currentGroupStatus.ToString("F0", CultureInfo.CurrentCulture.NumberFormat);                               
                textBoxStatus.BeginInvoke(
                   new Action(() =>
                   {
                       textBoxStatus.Text = strStatus;
                   }
                ));
                label_GroupStatusDescription.BeginInvoke(
                   new Action(() =>
                   {
                       label_GroupStatusDescription.Text = strGroupStatusDescription;
                   }
                ));
            
            }
            catch (Exception ex)
            {
                label_GroupStatusDescription.Text = "Exception in CurrentGroupStateHandlerChanged: " + ex.Message; // DEBUG
            }
        }

        private void ErrorMessageHandlerChanged(string Message)
        {            
            label_ErrorMessage.BeginInvoke(
               new Action(() =>
               {
                   label_ErrorMessage.Text = Message;
               }
            ));
        }

        public void UpdateGroupStatus()
        {
            try
            {
                int lastGroupState = m_CurrentGroupStatus;
                if (m_xpsInterfaceForPolling != null)
                {
                    string errorString = string.Empty;
                    int result = m_xpsInterfaceForPolling.GroupStatusGet(m_GroupName, out m_CurrentGroupStatus, out errorString);
                    if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                    {
                        m_CurrentGroupStatus = 0;
                        if (errorString.Length > 0)
                        {
                            int errorCode = 0;
                            int.TryParse(errorString, out errorCode);
                            m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                            m_ErrorMessageChanged(string.Format("GroupStatusGet ERROR {0}: {1}", result, m_errorDescription));
                        }
                        else
                            m_ErrorMessageChanged(string.Format("Communication failure with XPS after GroupStatusGet "));
                    }
                    else
                        result = m_xpsInterfaceForPolling.GroupStatusStringGet(m_CurrentGroupStatus, out m_CurrentGroupStatusDescription, out errorString);

                    if ((m_CurrentGroupStatus != lastGroupState) && m_CurrentGroupStateChanged != null)
                        m_CurrentGroupStateChanged(m_CurrentGroupStatus, m_CurrentGroupStatusDescription);
                }
            }
            catch (Exception ex)
            {
                m_ErrorMessageChanged("Exception in UpdateGroupStatus: " + ex.Message);
            }
        }

        public void UpdateCurrentPosition()
        {
            try
            {
                double lastCurrentPosition = m_CurrentPosition[0];
                if (m_xpsInterfaceForPolling != null)
                {
                    if (m_IsPositioner == true)
                    {
                        string errorString = string.Empty;
                        int result = m_xpsInterfaceForPolling.GroupPositionCurrentGet(m_PositionerName, out m_CurrentPosition, 1, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            m_CurrentPosition[0] = 0;
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                m_ErrorMessageChanged(string.Format("GroupPositionCurrentGet ERROR {0}: {1}", result, m_errorDescription));
                            }
                            else
                                m_ErrorMessageChanged(string.Format("Communication failure with XPS after GroupPositionCurrentGet "));
                        }
                    }

                    if ((m_CurrentPosition[0] != lastCurrentPosition) && m_CurrentPositionChanged != null)
                        m_CurrentPositionChanged(m_CurrentPosition);
                }
            }
            catch (Exception ex)
            {
                m_ErrorMessageChanged("Exception in UpdateCurrentPosition: " + ex.Message);
            }
        }

        public void StartPolling()
        {
            try
            {
                if (m_pollingFlag == false)
                {
                    m_pollingFlag = true; // Start polling

                    // Create thread and start it
                    m_PollingThread = new Thread(new ParameterizedThreadStart(poll));
                    m_PollingThread.IsBackground = true;
                    m_PollingThread.Start();
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in StartPolling: " + ex.Message;
            }
        }

        public void StopPolling()
        {
            try
            {
                m_pollingFlag = false; // Stop the polling
                if (m_PollingThread != null)
                    m_PollingThread.Abort();
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in StopPolling: " + ex.Message;
            }
        }

        public void poll(object obj)
        {
            String errString;
            double[] currPos= new double[1];
            currPos[0] = 1.0;
            try
            {
                while ((m_pollingFlag == true) && (m_CommunicationOK == true))
                {
                    UpdateGroupStatus();
                    UpdateCurrentPosition();
                    int result = m_xpsInterface.GPIOAnalogSet(new String[] { "GPIO4.DAC1" },currPos ,out errString);
                    if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                    {
                        if (errString.Length > 0)
                        {
                            int errorCode = 0;
                            int.TryParse(errString, out errorCode);
                            m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errString);
                            label_ErrorMessage.Text = string.Format("poll ERROR {0}: {1}", result, m_errorDescription);
                        }
                        else
                            label_ErrorMessage.Text = string.Format("Communication failure with XPS after GroupInitialize ");
                    }
                
                // Tempo in relation to the polling frequency
                Thread.Sleep(m_PollingInterval);
                }
            }
            catch (Exception ex)
            {
                m_ErrorMessageChanged("Exception in poll: " + ex.Message);
            }
        }

        /// <summary>
        /// Socket opening and start polling
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ConnectButton(object sender, EventArgs e)
        {
            // Get IP address and Ip port from form front panel
            m_IPAddress = textBox_IPAddress.Text;
            int.TryParse(textBox_IPPort.Text, out m_IPPort);

            m_PositionerName = TextBox_Group.Text;
            int index = m_PositionerName.LastIndexOf('.');
            if (index != -1)
            {
                m_IsPositioner = true;
                m_GroupName = m_PositionerName.Substring(0, index);
                label_ErrorMessage.Text = string.Empty;
            }
            else
            {
                m_IsPositioner = false;
                m_GroupName = m_PositionerName;
                label_ErrorMessage.Text = "Must be a positioner name not a group name";
            }

            label_GroupStatusDescription.Text = string.Empty;
            m_XPSControllerVersion = string.Empty;
            m_errorDescription = string.Empty;

            try
            {
                // Open socket #1 to order
                if (m_xpsInterface == null)
                    m_xpsInterface = new CommandInterfaceXPS.XPS();
                if (m_xpsInterface != null)
                {
                    // Open socket
                    int returnValue = m_xpsInterface.OpenInstrument(m_IPAddress, m_IPPort, DEFAULT_TIMEOUT);
                    if (returnValue == 0)
                    {
                        string errorString = string.Empty;
                        int result = m_xpsInterface.FirmwareVersionGet(out m_XPSControllerVersion, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                m_XPSControllerVersion = string.Format("FirmwareVersionGet ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                m_XPSControllerVersion = string.Format("Communication failure with XPS after FirmwareVersionGet ");
                        }
                        else
                        {

                            label_MessageCommunication.ForeColor = Color.Green;
                            label_MessageCommunication.Text = string.Format("Connected to XPS");
                            m_CommunicationOK = true;
                        }
                    }
                }
                else
                    m_XPSControllerVersion = "XPS instance is NULL";

                // Open socket #2 for polling
                if (m_xpsInterfaceForPolling == null)
                    m_xpsInterfaceForPolling = new CommandInterfaceXPS.XPS();
                if (m_xpsInterfaceForPolling != null)
                {
                    // Open socket
                    int returnValue = m_xpsInterfaceForPolling.OpenInstrument(m_IPAddress, m_IPPort, DEFAULT_TIMEOUT);
                    if (returnValue == 0)
                    {
                        string errorString = string.Empty;
                        int result = m_xpsInterfaceForPolling.FirmwareVersionGet(out m_XPSControllerVersion, out errorString);
                        if (result != CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                            StartPolling();
                    }
                }

                if (m_XPSControllerVersion.Length <= 0)
                    m_XPSControllerVersion = "No detected XPS";

                this.Text = string.Format("XPS Application - {0}", m_XPSControllerVersion);
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in ConnectButton: " + ex.Message;
            }
        }

        /// <summary>
        /// Stop polling and Close socket
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonDisconnect_Click(object sender, EventArgs e)
        {
            try
            {
                m_CommunicationOK = false;
                m_pollingFlag = false;

                if (m_xpsInterfaceForPolling != null)
                    m_xpsInterfaceForPolling.CloseInstrument();

                if (m_xpsInterface != null)
                    m_xpsInterface.CloseInstrument();
                
                label_MessageCommunication.ForeColor = Color.Red;
                label_MessageCommunication.Text = string.Format("Disconnected from XPS");
                label_ErrorMessage.Text = string.Empty;
                label_GroupStatusDescription.Text = string.Empty;
                m_XPSControllerVersion = string.Empty;
                m_errorDescription = string.Empty;
                this.Text = "XPS Application";
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonDisconnect_Click: " + ex.Message;
            }
        }

        /// <summary>
        /// Button to perform a GroupInitialize
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonInitialize_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_xpsInterface != null)
                {
                    string errorString = string.Empty;
                    int result = m_xpsInterface.GroupInitialize(m_GroupName, out errorString);
                    if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                    {
                        if (errorString.Length > 0)
                        {
                            int errorCode = 0;
                            int.TryParse(errorString, out errorCode);
                            m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                            label_ErrorMessage.Text = string.Format("GroupInitialize ERROR {0}: {1}", result, m_errorDescription);
                        }
                        else
                            label_ErrorMessage.Text = string.Format("Communication failure with XPS after GroupInitialize ");
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonInitialize_Click: " + ex.Message;
            }
        }

        /// <summary>
        /// Button to perform a group home search
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonHome_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_xpsInterface != null)
                {
                    string errorString = string.Empty;
                    int result = m_xpsInterface.GroupHomeSearch(m_GroupName, out errorString);
                    if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                    {
                        if (errorString.Length > 0)
                        {
                            int errorCode = 0;
                            int.TryParse(errorString, out errorCode);
                            m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                            label_ErrorMessage.Text = string.Format("GroupHomeSearch ERROR {0}: {1}", result, m_errorDescription);
                        }
                        else
                            label_ErrorMessage.Text = string.Format("Communication failure with XPS after GroupHomeSearch ");
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonHome_Click: " + ex.Message;
            }
        }

        /// <summary>
        /// Button to perform a group kill
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonKill_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_xpsInterface != null)
                {
                    string errorString = string.Empty;
                    int result = m_xpsInterface.GroupKill(m_GroupName, out errorString);
                    if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                    {
                        if (errorString.Length > 0)
                        {
                            int errorCode = 0;
                            int.TryParse(errorString, out errorCode);
                            m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                            label_ErrorMessage.Text = string.Format("GroupKill ERROR {0}: {1}", result, m_errorDescription);
                        }
                        else
                            label_ErrorMessage.Text = string.Format("Communication failure with XPS after GroupKill ");
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonKill_Click: " + ex.Message;
            }
        }
        /// <summary>
        /// Button to perform an absolute motion
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonMoveTo_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_IsPositioner == true)
                {
                    double.TryParse(textBoxTarget.Text, out m_TargetPosition[0]);
                    if ((m_xpsInterface != null) && (m_CommunicationOK == true))
                    {
                        string errorString = string.Empty;
                        int result = m_xpsInterface.GroupMoveAbsolute(m_PositionerName, m_TargetPosition, 1, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupMoveAbsolute ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after GroupMoveAbsolute ");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonMoveTo_Click: " + ex.Message;
            }
        }

        private void EnableTrkButton_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_IsPositioner == true)
                {

                    if ((m_xpsInterface != null) && (m_CommunicationOK == true))
                    {
                        string errorString = string.Empty;
                        string statusString = string.Empty;
                        int hardwareStatus = 0;
                        label_ErrorMessage.Text = "buttonEnableTrk: " ;
                        double trk_offset = 0.0;
                        double trk_scale = 10.0;
                        double trk_vel = 500.0;
                        double trk_acc = 2000.0;
                       // MessageBox.Show(m_PositionerName);
                        //MessageBox.Show(gpioname);
                        int result = m_xpsInterface.PositionerAnalogTrackingPositionParametersSet(m_PositionerName, gpioname, trk_offset, trk_scale, trk_vel, trk_acc, out errorString);



                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupTrkParam ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after Paramset ");
                        }
                        //MessageBox.Show(m_PositionerName);
                        String trackType = String.Format("Position");

                        //result = m_xpsInterface.PositionerHardwareStatusGet(m_PositionerName, out hardwareStatus, out errorString);
                        //result = m_xpsInterface.PositionerHardwareStatusStringGet(hardwareStatus, out statusString, out errorString);
                        //MessageBox.Show(statusString);

                        result = m_xpsInterface.GroupAnalogTrackingModeEnable(m_GroupName, trackType, out errorString);
                        //int result = m_xpsInterface.GroupMoveAbsolute(m_PositionerName, m_TargetPosition, 1, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupTrkEnable ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after EnableTrk ");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonEnableTrk: " + ex.Message;
            }
        }

        private void DisableTrkButton_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_IsPositioner == true)
                {

                    if ((m_xpsInterface != null) && (m_CommunicationOK == true))
                    {
                        string errorString = string.Empty;
                        label_ErrorMessage.Text = "buttonDisableTrk: ";
   
                       int result = m_xpsInterface.GroupAnalogTrackingModeDisable(m_GroupName, out errorString);
                        //int result = m_xpsInterface.GroupMoveAbsolute(m_PositionerName, m_TargetPosition, 1, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupMoveAbsolute ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after EnableTrk ");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonDisableTrk: " + ex.Message;
            }
        }

        private void StartAOButton_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_IsPositioner == true)
                {

                    if ((m_xpsInterface != null) && (m_CommunicationOK == true))
                    {
                        string errorString = string.Empty;
                        label_ErrorMessage.Text = "buttonDisableTrk: ";
                        //m_xpsInterface
                        String test;
                        String test2;
                        for (int i = 0; i < 8; i++)
                        {

                            m_xpsInterface.EventExtendedRemove(i, out errorString);
                        }
                        int result = m_xpsInterface.EventExtendedConfigurationTriggerSet(new string[] {"Always" }, new string[] { "1" }, new string[] { "0" }, new string[] { "0" }, new string[] { "0" }, out errorString);

                         result = m_xpsInterface.EventExtendedConfigurationActionSet(new string[] { "GPIO4.DAC1.DACSet.CurrentPosition" }, new string[] { m_PositionerName }, new string[] { "0.1" }, new string[] { "0" }, new string[] { "0" }, out errorString);

                        //result = m_xpsInterface.EventExtendedConfigurationActionGet(out test, out errorString);

                        //MessageBox.Show(test);
                        int eventID;

                        result = m_xpsInterface.EventExtendedStart(out eventID, out errorString);
                        //result = m_xpsInterface.EventExtendedAllGet( out test, out errorString);

                        // MessageBox.Show(eventID.ToString());

                        //int result = m_xpsInterface.EventExtendedAllGet(out test,out errorString);

                        //result = m_xpsInterface.EventExtendedGet(eventID, out test, out test2, out errorString);
                        //MessageBox.Show(test);
                        //MessageBox.Show(test2);
                        //int result = m_xpsInterface.EventAdd(m_PositionerName,"Always","1","GPIO4.DAC1.DACSet.CurrentPosition", m_PositionerName,"0.1","0", out errorString);
                        //int result = m_xpsInterface.EventAdd(m_PositionerName, "Always","0", "GPIO4.DAC1.DACSet.CurrentPosition", m_PositionerName, "0.1", "0", out errorString);
                        //int result = m_xpsInterface.EventAdd(m_PositionerName, "Always", "0", String.Format("GPIO3.DO.DOToggle"), "4", "0", "0", out errorString);

                        //int result = m_xpsInterface.GroupAnalogTrackingModeDisable(m_GroupName, out errorString);
                        //int result = m_xpsInterface.GroupMoveAbsolute(m_PositionerName, m_TargetPosition, 1, out errorString);
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupSetAO ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after EnableTrk ");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonDisableTrk: " + ex.Message;
            }
        }

        private void StopAOButton_Click(object sender, EventArgs e)
        {
            try
            {
                label_ErrorMessage.Text = string.Empty;
                if (m_CommunicationOK == false)
                    label_ErrorMessage.Text = string.Format("Not connected to XPS");

                if (m_IsPositioner == true)
                {

                    if ((m_xpsInterface != null) && (m_CommunicationOK == true))
                    {
                        string errorString = string.Empty;
                        label_ErrorMessage.Text = "buttonDisableTrk: ";
                        //m_xpsInterface
                        int result=0;

                        for (int i = 0; i < 8; i++)
                        {

                            result = m_xpsInterface.EventExtendedRemove(i, out errorString);
                        }
                        
                        if (result == CommandInterfaceXPS.XPS.FAILURE) // Communication failure with XPS 
                        {
                            if (errorString.Length > 0)
                            {
                                int errorCode = 0;
                                int.TryParse(errorString, out errorCode);
                                m_xpsInterface.ErrorStringGet(errorCode, out m_errorDescription, out errorString);
                                label_ErrorMessage.Text = string.Format("GroupSetAO ERROR {0}: {1}", result, m_errorDescription);
                            }
                            else
                                label_ErrorMessage.Text = string.Format("Communication failure with XPS after EnableTrk ");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                label_ErrorMessage.Text = "Exception in buttonDisableTrk: " + ex.Message;
            }
        }
    }
}