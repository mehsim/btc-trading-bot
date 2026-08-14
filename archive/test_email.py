#!/usr/bin/env python3
import os
import sys
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

def run_test():
    print("==================================================")
    print("          UBOTE SMTP EMAIL VERIFICATION           ")
    print("==================================================")

    # 1. Retrieve or prompt SMTP Host
    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        smtp_host = input("Enter SMTP Host (default: smtp.gmail.com): ").strip()
        if not smtp_host:
            smtp_host = "smtp.gmail.com"

    # 2. Retrieve or prompt SMTP Port
    smtp_port_str = os.getenv("SMTP_PORT")
    if not smtp_port_str:
        smtp_port_str = input("Enter SMTP Port (default: 587): ").strip()
        if not smtp_port_str:
            smtp_port_str = "587"
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"Invalid port: {smtp_port_str}. Defaulting to 587.")
        smtp_port = 587

    # 3. Retrieve or prompt SMTP User
    smtp_user = os.getenv("SMTP_USER")
    if not smtp_user:
        smtp_user = input("Enter SMTP User (your sender email address): ").strip()
        if not smtp_user:
            print("Error: SMTP User is required to authenticate. Aborting.")
            return

    # 4. Retrieve or prompt SMTP Password
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not smtp_password:
        smtp_password = input("Enter SMTP Password (or App Password): ").strip()
        if not smtp_password:
            print("Error: SMTP Password is required. Aborting.")
            return

    # 5. Recipient Email
    email_to = os.getenv("EMAIL_TO", "mehsimleo@gmail.com")
    print(f"\nAttempting to send test verification email to: {email_to}")
    print(f"Using SMTP Server: {smtp_host}:{smtp_port} as user {smtp_user}...")

    # 6. Compose HTML body
    subject = "🚀 [UBOTE Test Alert] SMTP Verification Successful!"
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px; max-width: 600px; margin: auto;">
        <h2 style="color: #00b0ff; border-bottom: 2px solid #161a22; padding-bottom: 10px;">✅ SMTP Connection Successful!</h2>
        <p>This is a test notification confirming that your UBOTE trading bot is successfully connected to your email provider.</p>
        <div style="background-color: #161a22; padding: 15px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #00b0ff;">
            <p style="margin: 4px 0;"><b>SMTP Host:</b> {smtp_host}</p>
            <p style="margin: 4px 0;"><b>SMTP Port:</b> {smtp_port}</p>
            <p style="margin: 4px 0;"><b>Sender User:</b> {smtp_user}</p>
            <p style="margin: 4px 0;"><b>Recipient:</b> {email_to}</p>
        </div>
        <p style="font-size: 11px; color: #8f9bb3;">You are now ready to receive real-time Take Profit alerts directly in your inbox.</p>
    </body>
    </html>
    """

    try:
        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email_to
        
        # Test connection
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            print("Connecting to server and starting TLS...")
            server.starttls()
            print("Logging in...")
            server.login(smtp_user, smtp_password)
            print("Sending message...")
            server.send_message(msg)
            print("\n🎉 SUCCESS! Test email sent successfully. Please check your inbox.")
            
            # Offer to save to .env
            save_env = input("\nWould you like to save these credentials to your local .env file? (y/n): ").strip().lower()
            if save_env == 'y':
                with open(".env", "a") as f:
                    f.write(f"\n# SMTP Email Credentials\n")
                    f.write(f"SMTP_HOST=\"{smtp_host}\"\n")
                    f.write(f"SMTP_PORT={smtp_port}\n")
                    f.write(f"SMTP_USER=\"{smtp_user}\"\n")
                    f.write(f"SMTP_PASSWORD=\"{smtp_password}\"\n")
                    f.write(f"EMAIL_TO=\"{email_to}\"\n")
                print("Credentials appended successfully to .env.")
    except Exception as e:
        print(f"\n❌ FAILED: SMTP connection error: {e}")
        print("\nTroubleshooting tips:")
        print("1. If using Gmail, make sure you generated and used an 'App Password', not your primary login password.")
        print("2. Check if your network/router blocks port 587 SMTP traffic.")
        print("3. Verify if your hosting server requires specific proxies.")

if __name__ == "__main__":
    run_test()
