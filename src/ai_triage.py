import pandas as pd
import random
import time

def generate_complaints(n=20):
    """
    Generate synthetic complaint texts.
    """
    card_fraud_templates = [
        "I see a charge of ${} that I did not make",
        "My card was used at a store in {} but I was at home",
        "Multiple small charges appeared on my statement overnight totaling ${}",
        "I got a notification for a ${} purchase I never authorized",
        "There is a mysterious transaction for ${} from a merchant I don't recognize",
        "Someone used my credit card details online for ${}",
        "A charge from {} for ${} just showed up, fraud!",
        "I still have my card, but someone spent ${} at a gas station",
        "Unauthorized charge of ${}, please cancel my card immediately",
        "Please help, my card was charged ${} while I was asleep"
    ]
    
    upi_failure_templates = [
        "My UPI payment of ₹{} failed but the amount was deducted from my account",
        "I sent money to the wrong VPA and need a refund of ₹{}",
        "Transaction of ₹{} stuck on pending for over 6 hours",
        "Payment of ₹{} failed due to server error but I was charged",
        "The merchant didn't receive my ₹{} UPI transfer but my bank debited it",
        "App crashed during UPI payment, ₹{} is missing from my account",
        "I scanned a QR code for ₹{} but it says payment failed, money deducted though",
        "Sent ₹{} to my friend but it's showing as pending since yesterday",
        "UPI app says network error but I got an SMS for ₹{} debit",
        "Double deduction of ₹{} for a single UPI payment"
    ]
    
    cities = ["New York", "London", "Miami", "Chicago", "Seattle"]
    merchants = ["Amazon", "Walmart", "Target", "Starbucks", "BestBuy"]
    
    complaints = []
    
    for i in range(n // 2):
        amount = round(random.uniform(5.0, 500.0), 2)
        if "{}" in card_fraud_templates[i % len(card_fraud_templates)]:
            if "store in" in card_fraud_templates[i % len(card_fraud_templates)]:
                text = card_fraud_templates[i % len(card_fraud_templates)].format(random.choice(cities))
            elif "charge from" in card_fraud_templates[i % len(card_fraud_templates)]:
                text = card_fraud_templates[i % len(card_fraud_templates)].format(random.choice(merchants), amount)
            else:
                text = card_fraud_templates[i % len(card_fraud_templates)].format(amount)
        else:
             text = card_fraud_templates[i % len(card_fraud_templates)]
             
        complaints.append({
            "complaint_id": f"C-{1000 + i}",
            "complaint_text": text,
            "complaint_type": "card_fraud"
        })
        
    for i in range(n // 2, n):
        amount = random.randint(100, 10000)
        text = upi_failure_templates[(i - n//2) % len(upi_failure_templates)].format(amount)
        complaints.append({
            "complaint_id": f"C-{1000 + i}",
            "complaint_text": text,
            "complaint_type": "upi_failure"
        })
        
    random.shuffle(complaints)
    return pd.DataFrame(complaints)


def triage_with_mock(complaints_df):
    """
    Fallback if no API key. Generate realistic mock triage responses.
    """
    results = []
    for _, row in complaints_df.iterrows():
        comp_type = row['complaint_type']
        text = row['complaint_text'].lower()
        
        if comp_type == 'card_fraud':
            category = 'fraud'
            priority = 'high'
            response = "We have received your fraud report and blocked your card immediately. Our team is investigating the unauthorized charge."
        else:
            if "wrong vpa" in text:
                category = 'user_error'
                priority = 'low'
                response = "We understand you sent money to the wrong VPA. Please contact the recipient directly or your bank to request a reversal."
            else:
                category = 'technical_failure'
                priority = 'medium'
                response = "We see your UPI payment failed but was debited. The amount should automatically refund to your account within 48 hours."
                
        results.append({
            "ai_category": category,
            "ai_priority": priority,
            "ai_response": response
        })
        
    res_df = pd.DataFrame(results, index=complaints_df.index)
    return pd.concat([complaints_df, res_df], axis=1)


def triage_with_gemini(complaints_df, api_key):
    """
    For each complaint, send to Gemini API (use model 'gemini-2.0-flash').
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("google.generativeai not installed. Falling back to mock triage.")
        return triage_with_mock(complaints_df)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    results = []
    
    for _, row in complaints_df.iterrows():
        prompt = f"""
You are a customer support triage system for a financial services company. Analyze the following customer complaint and provide:
1. Category: exactly one of [fraud, technical_failure, user_error]
2. Priority: exactly one of [high, medium, low]
3. Response: A professional 2-sentence first-response to the customer.

Complaint: {row['complaint_text']}

Respond in this exact format:
Category: [category]
Priority: [priority]
Response: [your 2-sentence response]
"""
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            category = 'unknown'
            priority = 'unknown'
            ai_response = 'Could not parse response.'
            
            for line in text.split('\n'):
                line = line.strip()
                if line.lower().startswith('category:'):
                    category = line.split(':', 1)[1].strip().lower().replace('[', '').replace(']', '')
                elif line.lower().startswith('priority:'):
                    priority = line.split(':', 1)[1].strip().lower().replace('[', '').replace(']', '')
                elif line.lower().startswith('response:'):
                    ai_response = line.split(':', 1)[1].strip().replace('[', '').replace(']', '')
                    
            results.append({
                "ai_category": category,
                "ai_priority": priority,
                "ai_response": ai_response
            })
            
            time.sleep(0.5) # rate limit
            
        except Exception as e:
            print(f"API Error: {e}")
            # Fallback for this single one to mock logic just to be safe
            if row['complaint_type'] == 'card_fraud':
                 results.append({"ai_category": "fraud", "ai_priority": "high", "ai_response": "Fallback mock response due to API error."})
            else:
                 results.append({"ai_category": "technical_failure", "ai_priority": "medium", "ai_response": "Fallback mock response due to API error."})
                 
    res_df = pd.DataFrame(results, index=complaints_df.index)
    return pd.concat([complaints_df, res_df], axis=1)


def run_triage(complaints_df=None, api_key=None):
    """
    Main entry point.
    """
    if complaints_df is None:
        complaints_df = generate_complaints(20)
        
    if api_key and api_key.strip():
        return triage_with_gemini(complaints_df, api_key)
    else:
        return triage_with_mock(complaints_df)
