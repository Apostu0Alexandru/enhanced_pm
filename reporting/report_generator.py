from fpdf import FPDF

class MaintenanceReport(FPDF):
    def generate(self, alerts, predictions):
        self.add_page()
        self.set_font("Arial", size=12)
        self.cell(200, 10, txt="Predictive Maintenance Report", ln=1, align="C")
        self.multi_cell(0, 10, f"Critical Alerts: {len(alerts)}\nPrediction Summary: {predictions[-10:]}")
        self.output("reports/maintenance_report.pdf")