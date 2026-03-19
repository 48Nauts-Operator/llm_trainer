"""
Business Template: NautSpot Email & Marketing
Specialized patterns for email classification and marketing automation.
"""

NAUTSPOT_EMAIL_TEMPLATE = {
    "sales_inquiry": {
        "patterns": [
            "I'm interested in your {service}",
            "Can you tell me about {service}?", 
            "What's included in your {service} package?",
            "I'd like a quote for {service}",
            "How much does {service} cost?",
            "We need help with {service}",
            "Looking for {service} provider",
            "Do you offer {service} solutions?"
        ],
        "services": [
            "CRM consulting", "automation setup", "email marketing",
            "sales process optimization", "lead generation", 
            "customer segmentation", "campaign management",
            "business intelligence", "data analysis"
        ],
        "response": {
            "action": "email_classify",
            "template": "sales_inquiry_response",
            "personalization_level": "high",
            "send_timing": "immediate",
            "risk_level": "low",
            "department": "sales"
        }
    },
    
    "demo_request": {
        "patterns": [
            "Can I see a demo of {product}?",
            "I'd like to schedule a demo",
            "Show me how {product} works",
            "Can we book a {product} demonstration?",
            "I want to see {product} in action",
            "Let's set up a demo call",
            "When can I see a {product} walkthrough?"
        ],
        "products": [
            "NautSpot", "the CRM", "your platform", "the system",
            "your solution", "the automation", "the dashboard"
        ],
        "response": {
            "action": "schedule_demo",
            "template": "demo_booking",
            "personalization_level": "high",
            "send_timing": "immediate",
            "risk_level": "low",
            "priority": "high"
        }
    },
    
    "support_request": {
        "patterns": [
            "I'm having trouble with {feature}",
            "The {feature} isn't working",
            "Help needed with {feature}",
            "Bug report: {feature}",
            "Issue with {feature}",
            "Can't get {feature} to work",
            "Error in {feature}"
        ],
        "features": [
            "email automation", "contact import", "campaign creation",
            "dashboard", "reporting", "integrations", "API",
            "mobile app", "data sync", "templates"
        ],
        "response": {
            "action": "assign_support",
            "template": "support_acknowledgment", 
            "personalization_level": "medium",
            "send_timing": "immediate",
            "risk_level": "medium",
            "sla": "24_hours"
        }
    },
    
    "feature_request": {
        "patterns": [
            "Can you add {feature} to the platform?",
            "Would love to see {feature}",
            "Feature request: {feature}",
            "Is {feature} on your roadmap?",
            "When will you support {feature}?",
            "Need {feature} functionality",
            "Missing {feature} feature"
        ],
        "features": [
            "SMS automation", "WhatsApp integration", "advanced analytics",
            "API webhooks", "custom fields", "bulk actions",
            "mobile notifications", "team collaboration", "white-labeling"
        ],
        "response": {
            "action": "product_feedback",
            "template": "feature_acknowledgment",
            "personalization_level": "medium", 
            "send_timing": "24_hours",
            "risk_level": "low",
            "forward_to": "product_team"
        }
    },
    
    "pricing_inquiry": {
        "patterns": [
            "What are your {plan} prices?",
            "How much is the {plan} plan?",
            "Do you offer {discount_type}?",
            "What's included in {plan}?",
            "Are there volume discounts?",
            "Can I get a custom quote?",
            "What's your pricing structure?"
        ],
        "plans": [
            "enterprise", "professional", "starter", "premium",
            "team", "individual", "custom"
        ],
        "discount_types": [
            "nonprofit discounts", "educational pricing", "startup credits",
            "annual discounts", "volume pricing"
        ],
        "response": {
            "action": "pricing_inquiry",
            "template": "pricing_response",
            "personalization_level": "high",
            "send_timing": "immediate",
            "risk_level": "low",
            "include_calendar": True
        }
    },
    
    "integration_question": {
        "patterns": [
            "Does NautSpot integrate with {platform}?",
            "How do I connect {platform} to NautSpot?",
            "API documentation for {platform}?",
            "Can I sync data from {platform}?",
            "Is there a {platform} connector?",
            "Webhook support for {platform}?"
        ],
        "platforms": [
            "Salesforce", "HubSpot", "Mailchimp", "Zapier",
            "Stripe", "PayPal", "Shopify", "WordPress",
            "Google Workspace", "Microsoft 365", "Slack", "Teams"
        ],
        "response": {
            "action": "integration_inquiry",
            "template": "integration_response",
            "personalization_level": "medium",
            "send_timing": "4_hours",
            "risk_level": "low",
            "technical_review": True
        }
    },
    
    "competitor_comparison": {
        "patterns": [
            "How does NautSpot compare to {competitor}?",
            "Why choose NautSpot over {competitor}?",
            "NautSpot vs {competitor} features",
            "Is NautSpot better than {competitor}?",
            "What's different about NautSpot and {competitor}?",
            "Switching from {competitor} to NautSpot"
        ],
        "competitors": [
            "HubSpot", "Salesforce", "Pipedrive", "Zoho",
            "ActiveCampaign", "Mailchimp", "ConvertKit", "Klaviyo"
        ],
        "response": {
            "action": "competitive_inquiry",
            "template": "comparison_response",
            "personalization_level": "high",
            "send_timing": "immediate",
            "risk_level": "medium",
            "competitive_intel": True
        }
    },
    
    "onboarding_question": {
        "patterns": [
            "How do I get started with {feature}?",
            "Setup help for {feature}",
            "Best practices for {feature}",
            "How to configure {feature}?",
            "Training on {feature}",
            "Getting started with NautSpot",
            "Onboarding assistance needed"
        ],
        "features": [
            "email campaigns", "contact management", "automation",
            "reporting", "integrations", "team setup", "data import"
        ],
        "response": {
            "action": "onboarding_support",
            "template": "onboarding_guidance",
            "personalization_level": "high",
            "send_timing": "immediate",
            "risk_level": "low",
            "include_resources": True
        }
    }
}

# Marketing automation patterns
NAUTSPOT_MARKETING_TEMPLATE = {
    "lead_scoring": {
        "patterns": [
            "Contact from {source} interested in {service}",
            "{company_type} company requesting {service} info",
            "Website visitor downloaded {resource}",
            "Email subscriber clicked {campaign} link",
            "Contact attended {event} webinar",
            "Demo request from {industry} company"
        ],
        "sources": [
            "website", "LinkedIn", "referral", "content download",
            "webinar", "demo", "trade show", "cold outreach"
        ],
        "services": [
            "automation", "CRM setup", "consulting", "training"
        ],
        "company_types": [
            "enterprise", "SMB", "startup", "agency", "nonprofit"
        ],
        "resources": [
            "whitepaper", "case study", "template", "guide"
        ],
        "events": [
            "product", "industry", "training"
        ],
        "industries": [
            "SaaS", "ecommerce", "consulting", "healthcare", "finance"
        ],
        "response": {
            "action": "lead_scoring",
            "score": "calculate_based_on_criteria",
            "segment": "determine_segment",
            "next_action": "assign_sequence",
            "priority": "medium"
        }
    },
    
    "customer_segmentation": {
        "patterns": [
            "Customer with {behavior} pattern",
            "User showing {engagement} engagement",
            "Contact in {industry} vertical",
            "Account with {size} employee count",
            "Lead from {traffic_source} source",
            "Subscriber with {activity} activity"
        ],
        "behaviors": [
            "high-value", "price-sensitive", "feature-focused",
            "support-heavy", "self-service", "enterprise-ready"
        ],
        "engagement": [
            "high", "medium", "low", "declining", "increasing"
        ],
        "industries": [
            "technology", "healthcare", "finance", "retail", "manufacturing"
        ],
        "sizes": [
            "1-10", "11-50", "51-200", "201-1000", "1000+"
        ],
        "traffic_sources": [
            "organic", "paid", "social", "referral", "direct", "email"
        ],
        "activities": [
            "high-open", "high-click", "low-engagement", "churning"
        ],
        "response": {
            "action": "customer_segmentation",
            "segment": "assign_segment",
            "campaign_type": "select_campaign",
            "messaging": "customize_messaging",
            "frequency": "optimize_frequency"
        }
    }
}

def get_email_template():
    """Return the NautSpot email template"""
    return NAUTSPOT_EMAIL_TEMPLATE

def get_marketing_template():
    """Return the NautSpot marketing template"""
    return NAUTSPOT_MARKETING_TEMPLATE

def get_combined_template():
    """Return combined email and marketing templates"""
    return {**NAUTSPOT_EMAIL_TEMPLATE, **NAUTSPOT_MARKETING_TEMPLATE}

if __name__ == "__main__":
    import json
    combined = get_combined_template()
    print(json.dumps(combined, indent=2))