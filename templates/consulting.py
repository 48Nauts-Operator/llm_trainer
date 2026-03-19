"""
Business Template: Consulting Firm
Email and inquiry classification patterns for consulting businesses.
"""

CONSULTING_TEMPLATE = {
    "sales_inquiry": {
        "patterns": [
            "I'm interested in your {service} services",
            "Can you help us with {service}?",
            "What are your rates for {service}?",
            "We need help with {service}",
            "I'd like to discuss {service} options",
            "Can you provide {service} consulting?",
            "We're looking for {service} expertise",
            "Do you offer {service} solutions?",
            "How much does {service} cost?",
            "We need a quote for {service}",
            "Can you send me information about {service}?",
            "I want to learn more about your {service} practice"
        ],
        "services": [
            "strategic planning", "digital transformation", "process optimization",
            "change management", "organizational design", "performance improvement",
            "business analysis", "project management", "leadership development",
            "operational excellence", "cost reduction", "revenue optimization",
            "market research", "competitive analysis", "due diligence",
            "risk management", "compliance", "governance"
        ],
        "response": {
            "priority": "high",
            "action": "personal_response",
            "department": "sales",
            "follow_up": "schedule_discovery_call"
        }
    },
    
    "support_request": {
        "patterns": [
            "I'm having trouble with {issue}",
            "The {system} isn't working as expected",
            "Can you help troubleshoot {issue}?",
            "We need technical support for {system}",
            "There's a problem with {issue}",
            "The {system} has stopped working",
            "Help needed with {issue}",
            "We're experiencing issues with {system}",
            "Something is wrong with {issue}",
            "The {system} is not performing correctly"
        ],
        "systems": [
            "implementation", "system", "process", "workflow", "solution",
            "platform", "integration", "configuration", "deployment",
            "dashboard", "reporting", "analytics", "automation"
        ],
        "issues": [
            "the new process", "our implementation", "the system integration",
            "the dashboard", "the reporting tool", "the workflow",
            "the automation", "the configuration", "the deployment"
        ],
        "response": {
            "priority": "medium",
            "action": "assign_support",
            "department": "support",
            "sla": "24_hours"
        }
    },
    
    "meeting_request": {
        "patterns": [
            "Can we schedule a call to discuss {topic}?",
            "Would you be available for a meeting about {topic}?",
            "I'd like to set up a consultation on {topic}",
            "Let's arrange a discovery call for {topic}",
            "When can we meet to discuss {topic}?",
            "I need a meeting about {topic}",
            "Can we book time to talk about {topic}?",
            "I'd like to schedule time to review {topic}",
            "Could we set up a call regarding {topic}?",
            "I want to discuss {topic} with your team"
        ],
        "topics": [
            "our project", "next steps", "implementation", "strategy",
            "requirements", "timeline", "budget", "proposal", "partnership",
            "the engagement", "deliverables", "scope", "approach",
            "objectives", "outcomes", "milestones", "resources"
        ],
        "response": {
            "priority": "high",
            "action": "schedule_meeting",
            "department": "sales",
            "meeting_type": "discovery_call"
        }
    },
    
    "project_update_request": {
        "patterns": [
            "What's the status of {project}?",
            "Can I get an update on {project}?",
            "How is {project} progressing?",
            "Where are we with {project}?",
            "I need a status update on {project}",
            "Can you provide an update on {project}?",
            "What's happening with {project}?",
            "Is {project} on track?"
        ],
        "projects": [
            "our project", "the implementation", "the engagement",
            "the strategic initiative", "the transformation",
            "the analysis", "the assessment", "the review"
        ],
        "response": {
            "priority": "medium",
            "action": "project_status_update",
            "department": "delivery",
            "requires_pm": True
        }
    },
    
    "contract_negotiation": {
        "patterns": [
            "I have questions about the {document}",
            "Can we discuss the {document} terms?",
            "I need clarification on the {document}",
            "The {document} needs some adjustments",
            "We'd like to modify the {document}",
            "Can we negotiate the {document}?",
            "I have concerns about the {document}"
        ],
        "documents": [
            "contract", "statement of work", "proposal", "agreement",
            "terms and conditions", "pricing", "scope", "timeline"
        ],
        "response": {
            "priority": "high",
            "action": "legal_review",
            "department": "sales",
            "requires_approval": True
        }
    },
    
    "referral_inquiry": {
        "patterns": [
            "{referrer} recommended your services",
            "{referrer} suggested I contact you",
            "I was referred by {referrer}",
            "{referrer} said you could help with {service}",
            "{referrer} spoke highly of your work",
            "I'm reaching out based on {referrer}'s recommendation"
        ],
        "referrers": [
            "John Smith", "a colleague", "my business partner",
            "a client", "another consultant", "my lawyer", "our board member"
        ],
        "services": [
            "strategic planning", "process improvement", "digital transformation"
        ],
        "response": {
            "priority": "high",
            "action": "personal_response",
            "department": "sales",
            "flag": "referral_source"
        }
    }
}

def get_template():
    """Return the consulting business template"""
    return CONSULTING_TEMPLATE

if __name__ == "__main__":
    import json
    print(json.dumps(CONSULTING_TEMPLATE, indent=2))