import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from typing import Optional


class AlertSystem:
    """Basic alerting system for data quality and model monitoring issues."""

    def __init__(self, smtp_server: Optional[str] = None, smtp_port: int = 587,
                 sender_email: Optional[str] = None, sender_password: Optional[str] = None,
                 recipient_emails: Optional[list] = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails or []

    def send_alert(self, subject: str, message: str, alert_type: str = "info"):
        """Send an alert via email or log it if email is not configured."""
        full_message = f"[{alert_type.upper()}] {message}"

        # Log the alert
        if alert_type == "error":
            logger.error(full_message)
        elif alert_type == "warning":
            logger.warning(full_message)
        else:
            logger.info(full_message)

        # Send email if configured
        if self.sender_email and self.recipient_emails:
            try:
                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = ', '.join(self.recipient_emails)
                msg['Subject'] = subject

                msg.attach(MIMEText(full_message, 'plain'))

                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.recipient_emails, text)
                server.quit()

                logger.info("Alert email sent successfully")
            except Exception as e:
                logger.error(f"Failed to send alert email: {e}")

    def alert_data_drift(self, drift_score: float, threshold: float):
        """Alert if data drift exceeds threshold."""
        if drift_score > threshold:
            self.send_alert(
                subject="Data Drift Detected",
                message=f"Data drift score {drift_score:.3f} exceeds threshold {threshold:.3f}",
                alert_type="warning"
            )

    def alert_data_quality_issue(self, issue_description: str):
        """Alert for data quality issues."""
        self.send_alert(
            subject="Data Quality Issue",
            message=issue_description,
            alert_type="error"
        )

    def alert_model_performance_drop(self, metric_name: str, current_value: float, baseline_value: float):
        """Alert if model performance drops significantly."""
        if current_value < baseline_value * 0.9:  # 10% drop threshold
            self.send_alert(
                subject="Model Performance Drop",
                message=f"{metric_name} dropped from {baseline_value:.3f} to {current_value:.3f}",
                alert_type="error"
            )
