namespace XPSApplicationTest
{
    partial class FormApplicationXPS
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.textBox_IPAddress = new System.Windows.Forms.TextBox();
            this.textBox_IPPort = new System.Windows.Forms.TextBox();
            this.labelIpAddress = new System.Windows.Forms.Label();
            this.labelIpPort = new System.Windows.Forms.Label();
            this.label_MessageCommunication = new System.Windows.Forms.Label();
            this.buttonConnect = new System.Windows.Forms.Button();
            this.TextBox_Group = new System.Windows.Forms.TextBox();
            this.labelGroup = new System.Windows.Forms.Label();
            this.label_ErrorMessage = new System.Windows.Forms.Label();
            this.buttonInitialize = new System.Windows.Forms.Button();
            this.buttonHome = new System.Windows.Forms.Button();
            this.buttonMoveTo = new System.Windows.Forms.Button();
            this.buttonKill = new System.Windows.Forms.Button();
            this.buttonDisconnect = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.stopAOVButton2 = new System.Windows.Forms.Button();
            this.startAOVButton2 = new System.Windows.Forms.Button();
            this.disableVTrkButton2 = new System.Windows.Forms.Button();
            this.enableVTrkButton2 = new System.Windows.Forms.Button();
            this.stopAOPButton2 = new System.Windows.Forms.Button();
            this.startAOPButton2 = new System.Windows.Forms.Button();
            this.disablePTrkButton2 = new System.Windows.Forms.Button();
            this.enablePTrkButton2 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.buttonKill2 = new System.Windows.Forms.Button();
            this.buttonMoveTo2 = new System.Windows.Forms.Button();
            this.buttonHome2 = new System.Windows.Forms.Button();
            this.buttonInitialize2 = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.textBoxTarget2 = new System.Windows.Forms.TextBox();
            this.textBoxPosition2 = new System.Windows.Forms.TextBox();
            this.textBoxStatus2 = new System.Windows.Forms.TextBox();
            this.TextBox_Group2 = new System.Windows.Forms.TextBox();
            this.stopAOVButton = new System.Windows.Forms.Button();
            this.startAOVButton = new System.Windows.Forms.Button();
            this.disableVTrkButton = new System.Windows.Forms.Button();
            this.enableVTrkButton = new System.Windows.Forms.Button();
            this.stopAOPButton = new System.Windows.Forms.Button();
            this.startAOPButton = new System.Windows.Forms.Button();
            this.disablePTrkButton = new System.Windows.Forms.Button();
            this.enablePTrkButton = new System.Windows.Forms.Button();
            this.label_GroupStatusDescription = new System.Windows.Forms.Label();
            this.labelPosition = new System.Windows.Forms.Label();
            this.labelStatus = new System.Windows.Forms.Label();
            this.textBoxTarget = new System.Windows.Forms.TextBox();
            this.textBoxPosition = new System.Windows.Forms.TextBox();
            this.textBoxStatus = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // textBox_IPAddress
            // 
            this.textBox_IPAddress.Location = new System.Drawing.Point(90, 21);
            this.textBox_IPAddress.Margin = new System.Windows.Forms.Padding(2);
            this.textBox_IPAddress.Name = "textBox_IPAddress";
            this.textBox_IPAddress.Size = new System.Drawing.Size(92, 20);
            this.textBox_IPAddress.TabIndex = 0;
            this.textBox_IPAddress.Text = "192.168.0.254";
            // 
            // textBox_IPPort
            // 
            this.textBox_IPPort.Location = new System.Drawing.Point(90, 46);
            this.textBox_IPPort.Margin = new System.Windows.Forms.Padding(2);
            this.textBox_IPPort.Name = "textBox_IPPort";
            this.textBox_IPPort.Size = new System.Drawing.Size(39, 20);
            this.textBox_IPPort.TabIndex = 0;
            this.textBox_IPPort.Text = "5001";
            // 
            // labelIpAddress
            // 
            this.labelIpAddress.AutoSize = true;
            this.labelIpAddress.Location = new System.Drawing.Point(20, 23);
            this.labelIpAddress.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelIpAddress.Name = "labelIpAddress";
            this.labelIpAddress.Size = new System.Drawing.Size(57, 13);
            this.labelIpAddress.TabIndex = 1;
            this.labelIpAddress.Text = "IP address";
            // 
            // labelIpPort
            // 
            this.labelIpPort.AutoSize = true;
            this.labelIpPort.Location = new System.Drawing.Point(20, 49);
            this.labelIpPort.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelIpPort.Name = "labelIpPort";
            this.labelIpPort.Size = new System.Drawing.Size(39, 13);
            this.labelIpPort.TabIndex = 1;
            this.labelIpPort.Text = "IP Port";
            // 
            // label_MessageCommunication
            // 
            this.label_MessageCommunication.AutoSize = true;
            this.label_MessageCommunication.Location = new System.Drawing.Point(11, 165);
            this.label_MessageCommunication.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label_MessageCommunication.Name = "label_MessageCommunication";
            this.label_MessageCommunication.Size = new System.Drawing.Size(0, 13);
            this.label_MessageCommunication.TabIndex = 1;
            // 
            // buttonConnect
            // 
            this.buttonConnect.Location = new System.Drawing.Point(90, 76);
            this.buttonConnect.Margin = new System.Windows.Forms.Padding(2);
            this.buttonConnect.Name = "buttonConnect";
            this.buttonConnect.Size = new System.Drawing.Size(75, 23);
            this.buttonConnect.TabIndex = 2;
            this.buttonConnect.Text = "Connect";
            this.buttonConnect.UseVisualStyleBackColor = true;
            this.buttonConnect.Click += new System.EventHandler(this.ConnectButton);
            // 
            // TextBox_Group
            // 
            this.TextBox_Group.Location = new System.Drawing.Point(16, 33);
            this.TextBox_Group.Name = "TextBox_Group";
            this.TextBox_Group.Size = new System.Drawing.Size(101, 20);
            this.TextBox_Group.TabIndex = 3;
            this.TextBox_Group.Text = "Group1.Pos";
            // 
            // labelGroup
            // 
            this.labelGroup.AutoSize = true;
            this.labelGroup.Location = new System.Drawing.Point(13, 17);
            this.labelGroup.Name = "labelGroup";
            this.labelGroup.Size = new System.Drawing.Size(82, 13);
            this.labelGroup.TabIndex = 4;
            this.labelGroup.Text = "Positioner name";
            // 
            // label_ErrorMessage
            // 
            this.label_ErrorMessage.AutoSize = true;
            this.label_ErrorMessage.ForeColor = System.Drawing.Color.Red;
            this.label_ErrorMessage.Location = new System.Drawing.Point(11, 213);
            this.label_ErrorMessage.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label_ErrorMessage.Name = "label_ErrorMessage";
            this.label_ErrorMessage.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.label_ErrorMessage.Size = new System.Drawing.Size(0, 13);
            this.label_ErrorMessage.TabIndex = 1;
            // 
            // buttonInitialize
            // 
            this.buttonInitialize.Location = new System.Drawing.Point(126, 77);
            this.buttonInitialize.Name = "buttonInitialize";
            this.buttonInitialize.Size = new System.Drawing.Size(75, 23);
            this.buttonInitialize.TabIndex = 5;
            this.buttonInitialize.Text = "Initialize";
            this.buttonInitialize.UseVisualStyleBackColor = true;
            this.buttonInitialize.Click += new System.EventHandler(this.buttonInitialize_Click);
            // 
            // buttonHome
            // 
            this.buttonHome.Location = new System.Drawing.Point(126, 105);
            this.buttonHome.Name = "buttonHome";
            this.buttonHome.Size = new System.Drawing.Size(75, 23);
            this.buttonHome.TabIndex = 5;
            this.buttonHome.Text = "Home";
            this.buttonHome.UseVisualStyleBackColor = true;
            this.buttonHome.Click += new System.EventHandler(this.buttonHome_Click);
            // 
            // buttonMoveTo
            // 
            this.buttonMoveTo.Location = new System.Drawing.Point(126, 133);
            this.buttonMoveTo.Name = "buttonMoveTo";
            this.buttonMoveTo.Size = new System.Drawing.Size(75, 23);
            this.buttonMoveTo.TabIndex = 5;
            this.buttonMoveTo.Text = "Move to";
            this.buttonMoveTo.UseVisualStyleBackColor = true;
            this.buttonMoveTo.Click += new System.EventHandler(this.buttonMoveTo_Click);
            // 
            // buttonKill
            // 
            this.buttonKill.Location = new System.Drawing.Point(126, 161);
            this.buttonKill.Name = "buttonKill";
            this.buttonKill.Size = new System.Drawing.Size(75, 23);
            this.buttonKill.TabIndex = 5;
            this.buttonKill.Text = "Kill";
            this.buttonKill.UseVisualStyleBackColor = true;
            this.buttonKill.Click += new System.EventHandler(this.buttonKill_Click);
            // 
            // buttonDisconnect
            // 
            this.buttonDisconnect.Location = new System.Drawing.Point(90, 103);
            this.buttonDisconnect.Margin = new System.Windows.Forms.Padding(2);
            this.buttonDisconnect.Name = "buttonDisconnect";
            this.buttonDisconnect.Size = new System.Drawing.Size(75, 23);
            this.buttonDisconnect.TabIndex = 2;
            this.buttonDisconnect.Text = "Disconnect";
            this.buttonDisconnect.UseVisualStyleBackColor = true;
            this.buttonDisconnect.Click += new System.EventHandler(this.buttonDisconnect_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.stopAOVButton2);
            this.groupBox1.Controls.Add(this.startAOVButton2);
            this.groupBox1.Controls.Add(this.disableVTrkButton2);
            this.groupBox1.Controls.Add(this.enableVTrkButton2);
            this.groupBox1.Controls.Add(this.stopAOPButton2);
            this.groupBox1.Controls.Add(this.startAOPButton2);
            this.groupBox1.Controls.Add(this.disablePTrkButton2);
            this.groupBox1.Controls.Add(this.enablePTrkButton2);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.buttonKill2);
            this.groupBox1.Controls.Add(this.buttonMoveTo2);
            this.groupBox1.Controls.Add(this.buttonHome2);
            this.groupBox1.Controls.Add(this.buttonInitialize2);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.textBoxTarget2);
            this.groupBox1.Controls.Add(this.textBoxPosition2);
            this.groupBox1.Controls.Add(this.textBoxStatus2);
            this.groupBox1.Controls.Add(this.TextBox_Group2);
            this.groupBox1.Controls.Add(this.stopAOVButton);
            this.groupBox1.Controls.Add(this.startAOVButton);
            this.groupBox1.Controls.Add(this.disableVTrkButton);
            this.groupBox1.Controls.Add(this.enableVTrkButton);
            this.groupBox1.Controls.Add(this.stopAOPButton);
            this.groupBox1.Controls.Add(this.startAOPButton);
            this.groupBox1.Controls.Add(this.disablePTrkButton);
            this.groupBox1.Controls.Add(this.enablePTrkButton);
            this.groupBox1.Controls.Add(this.label_GroupStatusDescription);
            this.groupBox1.Controls.Add(this.buttonKill);
            this.groupBox1.Controls.Add(this.buttonMoveTo);
            this.groupBox1.Controls.Add(this.buttonHome);
            this.groupBox1.Controls.Add(this.buttonInitialize);
            this.groupBox1.Controls.Add(this.labelPosition);
            this.groupBox1.Controls.Add(this.labelStatus);
            this.groupBox1.Controls.Add(this.labelGroup);
            this.groupBox1.Controls.Add(this.textBoxTarget);
            this.groupBox1.Controls.Add(this.textBoxPosition);
            this.groupBox1.Controls.Add(this.textBoxStatus);
            this.groupBox1.Controls.Add(this.TextBox_Group);
            this.groupBox1.Location = new System.Drawing.Point(498, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(614, 409);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "XPS";
            // 
            // stopAOVButton2
            // 
            this.stopAOVButton2.Location = new System.Drawing.Point(310, 270);
            this.stopAOVButton2.Name = "stopAOVButton2";
            this.stopAOVButton2.Size = new System.Drawing.Size(90, 21);
            this.stopAOVButton2.TabIndex = 34;
            this.stopAOVButton2.Text = "Stop Vel Read";
            this.stopAOVButton2.UseVisualStyleBackColor = true;
            // 
            // startAOVButton2
            // 
            this.startAOVButton2.Location = new System.Drawing.Point(310, 243);
            this.startAOVButton2.Name = "startAOVButton2";
            this.startAOVButton2.Size = new System.Drawing.Size(90, 21);
            this.startAOVButton2.TabIndex = 33;
            this.startAOVButton2.Text = "Start Vel Read";
            this.startAOVButton2.UseVisualStyleBackColor = true;
            // 
            // disableVTrkButton2
            // 
            this.disableVTrkButton2.Location = new System.Drawing.Point(310, 214);
            this.disableVTrkButton2.Name = "disableVTrkButton2";
            this.disableVTrkButton2.Size = new System.Drawing.Size(90, 23);
            this.disableVTrkButton2.TabIndex = 32;
            this.disableVTrkButton2.Text = "DisableVelTrk";
            this.disableVTrkButton2.UseVisualStyleBackColor = true;
            // 
            // enableVTrkButton2
            // 
            this.enableVTrkButton2.Location = new System.Drawing.Point(309, 186);
            this.enableVTrkButton2.Name = "enableVTrkButton2";
            this.enableVTrkButton2.Size = new System.Drawing.Size(91, 23);
            this.enableVTrkButton2.TabIndex = 31;
            this.enableVTrkButton2.Text = "EnableVelTrk";
            this.enableVTrkButton2.UseVisualStyleBackColor = true;
            // 
            // stopAOPButton2
            // 
            this.stopAOPButton2.Location = new System.Drawing.Point(310, 159);
            this.stopAOPButton2.Name = "stopAOPButton2";
            this.stopAOPButton2.Size = new System.Drawing.Size(90, 21);
            this.stopAOPButton2.TabIndex = 30;
            this.stopAOPButton2.Text = "Stop Pos Read";
            this.stopAOPButton2.UseVisualStyleBackColor = true;
            this.stopAOPButton2.Click += new System.EventHandler(this.StopAOPButton2_Click);
            // 
            // startAOPButton2
            // 
            this.startAOPButton2.Location = new System.Drawing.Point(310, 132);
            this.startAOPButton2.Name = "startAOPButton2";
            this.startAOPButton2.Size = new System.Drawing.Size(90, 21);
            this.startAOPButton2.TabIndex = 29;
            this.startAOPButton2.Text = "Start Pos Read";
            this.startAOPButton2.UseVisualStyleBackColor = true;
            this.startAOPButton2.Click += new System.EventHandler(this.StartAOPButton2_Click);
            // 
            // disablePTrkButton2
            // 
            this.disablePTrkButton2.Location = new System.Drawing.Point(310, 103);
            this.disablePTrkButton2.Name = "disablePTrkButton2";
            this.disablePTrkButton2.Size = new System.Drawing.Size(90, 23);
            this.disablePTrkButton2.TabIndex = 28;
            this.disablePTrkButton2.Text = "DisablePosTrk";
            this.disablePTrkButton2.UseVisualStyleBackColor = true;
            this.disablePTrkButton2.Click += new System.EventHandler(this.DisablePTrkButton2_Click);
            // 
            // enablePTrkButton2
            // 
            this.enablePTrkButton2.Location = new System.Drawing.Point(309, 75);
            this.enablePTrkButton2.Name = "enablePTrkButton2";
            this.enablePTrkButton2.Size = new System.Drawing.Size(91, 23);
            this.enablePTrkButton2.TabIndex = 27;
            this.enablePTrkButton2.Text = "EnablePosTrk";
            this.enablePTrkButton2.UseVisualStyleBackColor = true;
            this.enablePTrkButton2.Click += new System.EventHandler(this.EnablePTrkButton2_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.DimGray;
            this.label1.Location = new System.Drawing.Point(307, 57);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(37, 13);
            this.label1.TabIndex = 26;
            this.label1.Text = "Status";
            // 
            // buttonKill2
            // 
            this.buttonKill2.Location = new System.Drawing.Point(419, 160);
            this.buttonKill2.Name = "buttonKill2";
            this.buttonKill2.Size = new System.Drawing.Size(75, 23);
            this.buttonKill2.TabIndex = 25;
            this.buttonKill2.Text = "Kill";
            this.buttonKill2.UseVisualStyleBackColor = true;
            this.buttonKill2.Click += new System.EventHandler(this.ButtonKill2_Click);
            // 
            // buttonMoveTo2
            // 
            this.buttonMoveTo2.Location = new System.Drawing.Point(419, 132);
            this.buttonMoveTo2.Name = "buttonMoveTo2";
            this.buttonMoveTo2.Size = new System.Drawing.Size(75, 23);
            this.buttonMoveTo2.TabIndex = 24;
            this.buttonMoveTo2.Text = "Move to";
            this.buttonMoveTo2.UseVisualStyleBackColor = true;
            this.buttonMoveTo2.Click += new System.EventHandler(this.ButtonMoveTo2_Click);
            // 
            // buttonHome2
            // 
            this.buttonHome2.Location = new System.Drawing.Point(419, 104);
            this.buttonHome2.Name = "buttonHome2";
            this.buttonHome2.Size = new System.Drawing.Size(75, 23);
            this.buttonHome2.TabIndex = 23;
            this.buttonHome2.Text = "Home";
            this.buttonHome2.UseVisualStyleBackColor = true;
            this.buttonHome2.Click += new System.EventHandler(this.ButtonHome2_Click);
            // 
            // buttonInitialize2
            // 
            this.buttonInitialize2.Location = new System.Drawing.Point(419, 76);
            this.buttonInitialize2.Name = "buttonInitialize2";
            this.buttonInitialize2.Size = new System.Drawing.Size(75, 23);
            this.buttonInitialize2.TabIndex = 22;
            this.buttonInitialize2.Text = "Initialize";
            this.buttonInitialize2.UseVisualStyleBackColor = true;
            this.buttonInitialize2.Click += new System.EventHandler(this.ButtonInitialize2_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(503, 16);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 13);
            this.label2.TabIndex = 21;
            this.label2.Text = "Position";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(417, 16);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(37, 13);
            this.label3.TabIndex = 20;
            this.label3.Text = "Status";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(306, 16);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(82, 13);
            this.label4.TabIndex = 19;
            this.label4.Text = "Positioner name";
            // 
            // textBoxTarget2
            // 
            this.textBoxTarget2.Location = new System.Drawing.Point(505, 133);
            this.textBoxTarget2.Name = "textBoxTarget2";
            this.textBoxTarget2.Size = new System.Drawing.Size(75, 20);
            this.textBoxTarget2.TabIndex = 17;
            this.textBoxTarget2.Text = "0";
            this.textBoxTarget2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // textBoxPosition2
            // 
            this.textBoxPosition2.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxPosition2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxPosition2.Location = new System.Drawing.Point(506, 32);
            this.textBoxPosition2.Name = "textBoxPosition2";
            this.textBoxPosition2.ReadOnly = true;
            this.textBoxPosition2.Size = new System.Drawing.Size(75, 20);
            this.textBoxPosition2.TabIndex = 16;
            this.textBoxPosition2.Text = "0";
            this.textBoxPosition2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // textBoxStatus2
            // 
            this.textBoxStatus2.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxStatus2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxStatus2.Location = new System.Drawing.Point(420, 32);
            this.textBoxStatus2.Name = "textBoxStatus2";
            this.textBoxStatus2.ReadOnly = true;
            this.textBoxStatus2.Size = new System.Drawing.Size(75, 20);
            this.textBoxStatus2.TabIndex = 18;
            this.textBoxStatus2.Text = "0";
            this.textBoxStatus2.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // TextBox_Group2
            // 
            this.TextBox_Group2.Location = new System.Drawing.Point(309, 32);
            this.TextBox_Group2.Name = "TextBox_Group2";
            this.TextBox_Group2.Size = new System.Drawing.Size(101, 20);
            this.TextBox_Group2.TabIndex = 15;
            this.TextBox_Group2.Text = "Group2.Pos";
            // 
            // stopAOVButton
            // 
            this.stopAOVButton.Location = new System.Drawing.Point(17, 271);
            this.stopAOVButton.Name = "stopAOVButton";
            this.stopAOVButton.Size = new System.Drawing.Size(90, 21);
            this.stopAOVButton.TabIndex = 14;
            this.stopAOVButton.Text = "Stop Vel Read";
            this.stopAOVButton.UseVisualStyleBackColor = true;
            this.stopAOVButton.Click += new System.EventHandler(this.StopAOVButton_Click);
            // 
            // startAOVButton
            // 
            this.startAOVButton.Location = new System.Drawing.Point(17, 244);
            this.startAOVButton.Name = "startAOVButton";
            this.startAOVButton.Size = new System.Drawing.Size(90, 21);
            this.startAOVButton.TabIndex = 13;
            this.startAOVButton.Text = "Start Vel Read";
            this.startAOVButton.UseVisualStyleBackColor = true;
            this.startAOVButton.Click += new System.EventHandler(this.StartAOVButton_Click);
            // 
            // disableVTrkButton
            // 
            this.disableVTrkButton.Location = new System.Drawing.Point(17, 215);
            this.disableVTrkButton.Name = "disableVTrkButton";
            this.disableVTrkButton.Size = new System.Drawing.Size(90, 23);
            this.disableVTrkButton.TabIndex = 12;
            this.disableVTrkButton.Text = "DisableVelTrk";
            this.disableVTrkButton.UseVisualStyleBackColor = true;
            this.disableVTrkButton.Click += new System.EventHandler(this.DisableVTrkButton_Click);
            // 
            // enableVTrkButton
            // 
            this.enableVTrkButton.Location = new System.Drawing.Point(16, 187);
            this.enableVTrkButton.Name = "enableVTrkButton";
            this.enableVTrkButton.Size = new System.Drawing.Size(91, 23);
            this.enableVTrkButton.TabIndex = 11;
            this.enableVTrkButton.Text = "EnableVelTrk";
            this.enableVTrkButton.UseVisualStyleBackColor = true;
            this.enableVTrkButton.Click += new System.EventHandler(this.EnableVTrkButton_Click);
            // 
            // stopAOPButton
            // 
            this.stopAOPButton.Location = new System.Drawing.Point(17, 160);
            this.stopAOPButton.Name = "stopAOPButton";
            this.stopAOPButton.Size = new System.Drawing.Size(90, 21);
            this.stopAOPButton.TabIndex = 10;
            this.stopAOPButton.Text = "Stop Pos Read";
            this.stopAOPButton.UseVisualStyleBackColor = true;
            this.stopAOPButton.Click += new System.EventHandler(this.StopAOButton_Click);
            // 
            // startAOPButton
            // 
            this.startAOPButton.Location = new System.Drawing.Point(17, 133);
            this.startAOPButton.Name = "startAOPButton";
            this.startAOPButton.Size = new System.Drawing.Size(90, 21);
            this.startAOPButton.TabIndex = 9;
            this.startAOPButton.Text = "Start Pos Read";
            this.startAOPButton.UseVisualStyleBackColor = true;
            this.startAOPButton.Click += new System.EventHandler(this.StartAOButton_Click);
            // 
            // disablePTrkButton
            // 
            this.disablePTrkButton.Location = new System.Drawing.Point(17, 104);
            this.disablePTrkButton.Name = "disablePTrkButton";
            this.disablePTrkButton.Size = new System.Drawing.Size(90, 23);
            this.disablePTrkButton.TabIndex = 8;
            this.disablePTrkButton.Text = "DisablePosTrk";
            this.disablePTrkButton.UseVisualStyleBackColor = true;
            this.disablePTrkButton.Click += new System.EventHandler(this.DisableTrkButton_Click);
            // 
            // enablePTrkButton
            // 
            this.enablePTrkButton.Location = new System.Drawing.Point(16, 76);
            this.enablePTrkButton.Name = "enablePTrkButton";
            this.enablePTrkButton.Size = new System.Drawing.Size(91, 23);
            this.enablePTrkButton.TabIndex = 7;
            this.enablePTrkButton.Text = "EnablePosTrk";
            this.enablePTrkButton.UseVisualStyleBackColor = true;
            this.enablePTrkButton.Click += new System.EventHandler(this.EnableTrkButton_Click);
            // 
            // label_GroupStatusDescription
            // 
            this.label_GroupStatusDescription.AutoSize = true;
            this.label_GroupStatusDescription.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label_GroupStatusDescription.ForeColor = System.Drawing.Color.DimGray;
            this.label_GroupStatusDescription.Location = new System.Drawing.Point(14, 58);
            this.label_GroupStatusDescription.Name = "label_GroupStatusDescription";
            this.label_GroupStatusDescription.Size = new System.Drawing.Size(37, 13);
            this.label_GroupStatusDescription.TabIndex = 6;
            this.label_GroupStatusDescription.Text = "Status";
            // 
            // labelPosition
            // 
            this.labelPosition.AutoSize = true;
            this.labelPosition.Location = new System.Drawing.Point(210, 17);
            this.labelPosition.Name = "labelPosition";
            this.labelPosition.Size = new System.Drawing.Size(44, 13);
            this.labelPosition.TabIndex = 4;
            this.labelPosition.Text = "Position";
            // 
            // labelStatus
            // 
            this.labelStatus.AutoSize = true;
            this.labelStatus.Location = new System.Drawing.Point(124, 17);
            this.labelStatus.Name = "labelStatus";
            this.labelStatus.Size = new System.Drawing.Size(37, 13);
            this.labelStatus.TabIndex = 4;
            this.labelStatus.Text = "Status";
            // 
            // textBoxTarget
            // 
            this.textBoxTarget.Location = new System.Drawing.Point(212, 134);
            this.textBoxTarget.Name = "textBoxTarget";
            this.textBoxTarget.Size = new System.Drawing.Size(75, 20);
            this.textBoxTarget.TabIndex = 3;
            this.textBoxTarget.Text = "0";
            this.textBoxTarget.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // textBoxPosition
            // 
            this.textBoxPosition.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxPosition.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxPosition.Location = new System.Drawing.Point(213, 33);
            this.textBoxPosition.Name = "textBoxPosition";
            this.textBoxPosition.ReadOnly = true;
            this.textBoxPosition.Size = new System.Drawing.Size(75, 20);
            this.textBoxPosition.TabIndex = 3;
            this.textBoxPosition.Text = "0";
            this.textBoxPosition.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // textBoxStatus
            // 
            this.textBoxStatus.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxStatus.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxStatus.Location = new System.Drawing.Point(127, 33);
            this.textBoxStatus.Name = "textBoxStatus";
            this.textBoxStatus.ReadOnly = true;
            this.textBoxStatus.Size = new System.Drawing.Size(75, 20);
            this.textBoxStatus.TabIndex = 3;
            this.textBoxStatus.Text = "0";
            this.textBoxStatus.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.buttonDisconnect);
            this.groupBox2.Controls.Add(this.buttonConnect);
            this.groupBox2.Controls.Add(this.labelIpPort);
            this.groupBox2.Controls.Add(this.labelIpAddress);
            this.groupBox2.Controls.Add(this.textBox_IPPort);
            this.groupBox2.Controls.Add(this.textBox_IPAddress);
            this.groupBox2.Location = new System.Drawing.Point(9, 12);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(197, 141);
            this.groupBox2.TabIndex = 7;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "TCP IP";
            // 
            // FormApplicationXPS
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1192, 424);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.label_MessageCommunication);
            this.Controls.Add(this.label_ErrorMessage);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "FormApplicationXPS";
            this.Text = "XPS Application";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox textBox_IPAddress;
        private System.Windows.Forms.TextBox textBox_IPPort;
        private System.Windows.Forms.Label labelIpAddress;
        private System.Windows.Forms.Label labelIpPort;
        private System.Windows.Forms.Label label_MessageCommunication;
        private System.Windows.Forms.Button buttonConnect;
        private System.Windows.Forms.TextBox TextBox_Group;
        private System.Windows.Forms.Label labelGroup;
        private System.Windows.Forms.Label label_ErrorMessage;
        private System.Windows.Forms.Button buttonInitialize;
        private System.Windows.Forms.Button buttonHome;
        private System.Windows.Forms.Button buttonMoveTo;
        private System.Windows.Forms.Button buttonKill;
        private System.Windows.Forms.Button buttonDisconnect;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label labelPosition;
        private System.Windows.Forms.Label labelStatus;
        private System.Windows.Forms.TextBox textBoxTarget;
        private System.Windows.Forms.TextBox textBoxPosition;
        private System.Windows.Forms.TextBox textBoxStatus;
        private System.Windows.Forms.Label label_GroupStatusDescription;
        private System.Windows.Forms.Button disablePTrkButton;
        private System.Windows.Forms.Button enablePTrkButton;
        private System.Windows.Forms.Button stopAOPButton;
        private System.Windows.Forms.Button startAOPButton;
        private System.Windows.Forms.Button stopAOVButton;
        private System.Windows.Forms.Button startAOVButton;
        private System.Windows.Forms.Button disableVTrkButton;
        private System.Windows.Forms.Button enableVTrkButton;
        private System.Windows.Forms.Button stopAOVButton2;
        private System.Windows.Forms.Button startAOVButton2;
        private System.Windows.Forms.Button disableVTrkButton2;
        private System.Windows.Forms.Button enableVTrkButton2;
        private System.Windows.Forms.Button stopAOPButton2;
        private System.Windows.Forms.Button startAOPButton2;
        private System.Windows.Forms.Button disablePTrkButton2;
        private System.Windows.Forms.Button enablePTrkButton2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button buttonKill2;
        private System.Windows.Forms.Button buttonMoveTo2;
        private System.Windows.Forms.Button buttonHome2;
        private System.Windows.Forms.Button buttonInitialize2;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox textBoxTarget2;
        private System.Windows.Forms.TextBox textBoxPosition2;
        private System.Windows.Forms.TextBox textBoxStatus2;
        private System.Windows.Forms.TextBox TextBox_Group2;
    }
}

