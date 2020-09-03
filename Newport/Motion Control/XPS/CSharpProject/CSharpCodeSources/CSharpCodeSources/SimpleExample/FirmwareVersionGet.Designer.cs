namespace SimpleExample
{
    partial class FirmwareVersionGet
    {
        /// <summary>
        /// Variable nécessaire au concepteur.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Nettoyage des ressources utilisées.
        /// </summary>
        /// <param name="disposing">true si les ressources managées doivent être supprimées ; sinon, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Code généré par le Concepteur Windows Form

        /// <summary>
        /// Méthode requise pour la prise en charge du concepteur - ne modifiez pas
        /// le contenu de cette méthode avec l'éditeur de code.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.IPAddressTextBox = new System.Windows.Forms.TextBox();
            this.connectButtonButton = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.firmwareVersionLabel = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.errorMsgLabel = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(22, 27);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(75, 17);
            this.label1.TabIndex = 0;
            this.label1.Text = "IP address";
            // 
            // IPAddressTextBox
            // 
            this.IPAddressTextBox.Location = new System.Drawing.Point(107, 27);
            this.IPAddressTextBox.Name = "IPAddressTextBox";
            this.IPAddressTextBox.Size = new System.Drawing.Size(140, 22);
            this.IPAddressTextBox.TabIndex = 1;
            // 
            // connectButtonButton
            // 
            this.connectButtonButton.Location = new System.Drawing.Point(305, 21);
            this.connectButtonButton.Name = "connectButtonButton";
            this.connectButtonButton.Size = new System.Drawing.Size(102, 34);
            this.connectButtonButton.TabIndex = 2;
            this.connectButtonButton.Text = "Connect";
            this.connectButtonButton.UseVisualStyleBackColor = true;
            this.connectButtonButton.Click += new System.EventHandler(this.connectButtonButton_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(22, 71);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(119, 17);
            this.label2.TabIndex = 3;
            this.label2.Text = "Firmware version:";
            // 
            // firmwareVersionLabel
            // 
            this.firmwareVersionLabel.Location = new System.Drawing.Point(147, 71);
            this.firmwareVersionLabel.Name = "firmwareVersionLabel";
            this.firmwareVersionLabel.Size = new System.Drawing.Size(291, 41);
            this.firmwareVersionLabel.TabIndex = 4;
            this.firmwareVersionLabel.Text = "Vx.x.x";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(22, 129);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(44, 17);
            this.label3.TabIndex = 5;
            this.label3.Text = "Error:";
            // 
            // errorMsgLabel
            // 
            this.errorMsgLabel.AutoSize = true;
            this.errorMsgLabel.Location = new System.Drawing.Point(72, 129);
            this.errorMsgLabel.Name = "errorMsgLabel";
            this.errorMsgLabel.Size = new System.Drawing.Size(132, 17);
            this.errorMsgLabel.TabIndex = 6;
            this.errorMsgLabel.Text = "Returned message.";
            // 
            // FirmwareVersionGet
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(450, 167);
            this.Controls.Add(this.errorMsgLabel);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.firmwareVersionLabel);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.connectButtonButton);
            this.Controls.Add(this.IPAddressTextBox);
            this.Controls.Add(this.label1);
            this.Name = "FirmwareVersionGet";
            this.Text = "Get firmware version";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox IPAddressTextBox;
        private System.Windows.Forms.Button connectButtonButton;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label firmwareVersionLabel;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label errorMsgLabel;
    }
}

