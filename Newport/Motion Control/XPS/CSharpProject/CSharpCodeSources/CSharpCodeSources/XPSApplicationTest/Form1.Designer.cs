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
            this.textBox_IPAddress.Text = "192.168.33.229";
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
            this.groupBox1.Location = new System.Drawing.Point(222, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(300, 195);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "XPS";
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
            this.textBoxTarget.Text = "10";
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
            this.ClientSize = new System.Drawing.Size(534, 231);
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
    }
}

