"""Salesforce CRM Integration (Simulated)"""
import random
import time
from datetime import datetime, timedelta


class SalesforceCRM:
    """Simulated Salesforce CRM system"""
    
    def __init__(self):
        self.api_delay = 0.2  # Simulate API latency
    
    def get_account_info(self, vendor_name: str) -> dict:
        """Get account information from CRM"""
        time.sleep(self.api_delay)
        
        # Simulate CRM data
        return {
            "account_name": vendor_name,
            "account_tier": random.choice(["Enterprise", "Strategic", "Standard"]),
            "relationship_status": random.choice(["Excellent", "Good", "New"]),
            "lifetime_value": random.randint(100000, 5000000),
            "account_owner": "Sarah Chen",
            "last_interaction": (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
            "open_opportunities": random.randint(1, 5),
            "source": "Salesforce CRM"
        }
    
    def get_past_deals(self, vendor_name: str) -> dict:
        """Get historical deal data"""
        time.sleep(self.api_delay)
        
        num_deals = random.randint(1, 4)
        deals = []
        
        for i in range(num_deals):
            year = 2024 - i
            deals.append({
                "year": year,
                "deal_size": random.randint(50000, 500000),
                "price_per_seat": round(random.uniform(8, 15), 2),
                "seats": random.randint(100, 500),
                "contract_length": random.choice([1, 2, 3]),
                "discount_percent": round(random.uniform(10, 30), 1),
                "status": "Closed Won"
            })
        
        return {
            "vendor": vendor_name,
            "total_deals": num_deals,
            "deals": deals,
            "average_discount": round(sum(d["discount_percent"] for d in deals) / num_deals, 1),
            "source": "Salesforce CRM"
        }
    
    def get_competitor_intel(self, vendor_name: str) -> dict:
        """Get competitive intelligence from CRM"""
        time.sleep(self.api_delay)
        
        competitors = {
            "Slack": ["Microsoft Teams", "Zoom", "Google Chat"],
            "Datadog": ["New Relic", "Splunk", "Dynatrace"],
            "Snowflake": ["Databricks", "BigQuery", "Redshift"],
        }
        
        competitor_list = competitors.get(vendor_name, ["Competitor A", "Competitor B"])
        
        return {
            "primary_competitors": competitor_list[:2],
            "win_rate_vs_competitors": round(random.uniform(0.4, 0.7), 2),
            "common_objections": [
                "Price too high",
                "Competitor has better features",
                "Long contract commitment"
            ],
            "recommended_strategy": "Emphasize ROI and reference customer success stories",
            "source": "Salesforce CRM"
        }
    
    def update_opportunity(self, vendor_name: str, stage: str, deal_value: float) -> dict:
        """Update opportunity in CRM"""
        time.sleep(self.api_delay)
        
        return {
            "status": "success",
            "opportunity_id": f"OPP-{random.randint(10000, 99999)}",
            "vendor": vendor_name,
            "stage": stage,
            "deal_value": deal_value,
            "updated_at": datetime.now().isoformat(),
            "source": "Salesforce CRM"
        }
