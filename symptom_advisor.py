import json
import urllib.request
import urllib.parse

# ── Comprehensive Built-in Symptom → Medicine Knowledge Base ──────────────
SYMPTOM_DB = {
    "fever": {
        "medicines": ["Paracetamol (Acetaminophen)", "Ibuprofen", "Aspirin"],
        "dosage": "Paracetamol 500mg every 4-6 hours (max 4g/day). Ibuprofen 200-400mg every 6-8 hours.",
        "precautions": "Stay hydrated. Rest. Seek medical help if fever exceeds 103°F (39.4°C) or persists >3 days.",
        "specialist": "General Physician / Internal Medicine"
    },
    "headache": {
        "medicines": ["Paracetamol", "Ibuprofen", "Aspirin", "Sumatriptan (for migraines)"],
        "dosage": "Paracetamol 500-1000mg. Ibuprofen 200-400mg. Sumatriptan 50-100mg for migraines.",
        "precautions": "Avoid prolonged screen time. Stay hydrated. Seek help if sudden severe headache or with vision changes.",
        "specialist": "Neurologist"
    },
    "cold": {
        "medicines": ["Cetirizine", "Pseudoephedrine", "Dextromethorphan", "Paracetamol"],
        "dosage": "Cetirizine 10mg daily. Pseudoephedrine 60mg every 4-6h. Dextromethorphan 10-20mg every 4h.",
        "precautions": "Warm fluids, rest, and vitamin C. Avoid cold exposure. See doctor if symptoms persist >10 days.",
        "specialist": "General Physician / ENT Specialist"
    },
    "cough": {
        "medicines": ["Dextromethorphan", "Guaifenesin", "Ambroxol", "Honey with warm water"],
        "dosage": "Dextromethorphan 10-20mg every 4-6h. Guaifenesin 200-400mg every 4h.",
        "precautions": "Avoid smoking. Steam inhalation helps. See doctor if blood in sputum or cough >3 weeks.",
        "specialist": "Pulmonologist / General Physician"
    },
    "stomach pain": {
        "medicines": ["Omeprazole", "Ranitidine", "Antacid (Aluminum Hydroxide)", "Dicyclomine"],
        "dosage": "Omeprazole 20mg before meals. Antacid 10-20ml after meals.",
        "precautions": "Avoid spicy and oily food. Eat small frequent meals. See doctor if severe or persistent pain.",
        "specialist": "Gastroenterologist"
    },
    "diarrhea": {
        "medicines": ["ORS (Oral Rehydration Salt)", "Loperamide", "Zinc Supplements", "Probiotics"],
        "dosage": "ORS after each loose stool. Loperamide 2mg initial then 1mg per stool (max 8mg/day).",
        "precautions": "Maintain fluid intake. BRAT diet (Bananas, Rice, Applesauce, Toast). See doctor if bloody stool.",
        "specialist": "Gastroenterologist / General Physician"
    },
    "body pain": {
        "medicines": ["Ibuprofen", "Diclofenac", "Paracetamol", "Methyl Salicylate cream (topical)"],
        "dosage": "Ibuprofen 400mg every 8h. Diclofenac 50mg every 8h. Topical cream 2-3 times daily.",
        "precautions": "Rest affected area. Warm compress. See doctor if pain worsens or persists >7 days.",
        "specialist": "Orthopedist / Rheumatologist"
    },
    "allergy": {
        "medicines": ["Cetirizine", "Loratadine", "Fexofenadine", "Montelukast"],
        "dosage": "Cetirizine 10mg daily. Loratadine 10mg daily. Fexofenadine 120-180mg daily.",
        "precautions": "Avoid known allergens. Keep antihistamines accessible. See doctor for severe allergic reactions.",
        "specialist": "Allergist / Immunologist"
    },
    "back pain": {
        "medicines": ["Ibuprofen", "Diclofenac", "Muscle relaxants (Methocarbamol)", "Topical analgesic"],
        "dosage": "Ibuprofen 400mg every 8h. Methocarbamol 500mg 3-4 times daily.",
        "precautions": "Maintain good posture. Avoid heavy lifting. Physical therapy recommended. MRI if chronic.",
        "specialist": "Orthopedist / Spine Specialist"
    },
    "sore throat": {
        "medicines": ["Paracetamol", "Lozenges (Benzocaine)", "Warm salt water gargle", "Ibuprofen"],
        "dosage": "Paracetamol 500mg every 6h. Gargle with warm salt water 3-4 times daily.",
        "precautions": "Stay hydrated. Avoid irritants like smoke. See doctor if strep throat suspected.",
        "specialist": "ENT Specialist / General Physician"
    },
    "vomiting": {
        "medicines": ["Ondansetron", "Domperidone", "ORS", "Metoclopramide"],
        "dosage": "Ondansetron 4-8mg every 8h. Domperidone 10mg before meals.",
        "precautions": "Sip fluids slowly. Avoid solid foods initially. Seek ER if severe dehydration.",
        "specialist": "Gastroenterologist / Emergency Medicine"
    },
    "chest pain": {
        "medicines": ["Aspirin (if cardiac suspected)", "Nitroglycerin (prescribed)", "Antacids (if GERD)"],
        "dosage": "Aspirin 325mg chewed immediately if heart attack suspected. Seek immediate emergency care.",
        "precautions": "⚠️ CALL EMERGENCY SERVICES IMMEDIATELY if chest pain with shortness of breath, sweating, or jaw pain.",
        "specialist": "Cardiologist / Emergency Medicine"
    },
    "diabetes": {
        "medicines": ["Metformin", "Glimepiride", "Insulin (as prescribed)", "Sitagliptin"],
        "dosage": "Metformin 500-1000mg twice daily with meals. Dosage must be set by physician.",
        "precautions": "Regular blood glucose monitoring. Diet control. Exercise. Regular HbA1c tests.",
        "specialist": "Endocrinologist / Diabetologist"
    },
    "high blood pressure": {
        "medicines": ["Amlodipine", "Losartan", "Metoprolol", "Hydrochlorothiazide"],
        "dosage": "Amlodipine 5-10mg daily. Losartan 50-100mg daily. Must be prescribed by physician.",
        "precautions": "Reduce salt intake. Regular BP monitoring. Exercise. Avoid stress.",
        "specialist": "Cardiologist / Internal Medicine"
    },
    "anxiety": {
        "medicines": ["Alprazolam (prescribed)", "Buspirone", "SSRIs (Sertraline, Escitalopram)", "Propranolol"],
        "dosage": "These medications require psychiatric evaluation and prescription. Do not self-medicate.",
        "precautions": "Cognitive behavioral therapy recommended. Regular exercise. Avoid caffeine and alcohol.",
        "specialist": "Psychiatrist / Psychologist"
    },
    "skin rash": {
        "medicines": ["Hydrocortisone cream", "Calamine lotion", "Cetirizine", "Antifungal cream (Clotrimazole)"],
        "dosage": "Hydrocortisone 1% cream twice daily. Calamine lotion as needed. Cetirizine 10mg daily.",
        "precautions": "Keep area clean and dry. Avoid scratching. See dermatologist if spreading or infected.",
        "specialist": "Dermatologist"
    }
}


def search_openfda_drug(query):
    """Search OpenFDA drug labeling API for drug indications matching a symptom query."""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.fda.gov/drug/label.json?search=indications_and_usage:{encoded_query}&limit=5"
        req = urllib.request.Request(url, headers={'User-Agent': 'MediVisionAI/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = []
            for item in data.get('results', []):
                brand = item.get('openfda', {}).get('brand_name', ['Unknown'])[0]
                generic = item.get('openfda', {}).get('generic_name', ['N/A'])[0]
                indication = item.get('indications_and_usage', ['N/A'])[0][:400]
                dosage = item.get('dosage_and_administration', ['Refer to packaging'])[0][:400]
                warnings = item.get('warnings', ['No specific warnings listed'])[0][:300]
                results.append({
                    'brand_name': brand,
                    'generic_name': generic,
                    'indication': indication,
                    'dosage': dosage,
                    'warnings': warnings
                })
            return results
    except Exception as e:
        return []


def get_symptom_advice(user_query):
    """Matches user symptom query to local DB and supplements with OpenFDA data."""
    query_lower = user_query.lower().strip()

    # 1. Local DB match
    local_match = None
    for symptom_key, info in SYMPTOM_DB.items():
        if symptom_key in query_lower:
            local_match = (symptom_key, info)
            break

    # 2. OpenFDA API lookup
    fda_results = search_openfda_drug(user_query)

    # 3. Build response
    output = ""

    if local_match:
        key, info = local_match
        output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"  🩺 SYMPTOM IDENTIFIED: {key.upper()}\n"
        output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        output += f"💊 SUGGESTED MEDICINES:\n"
        for i, med in enumerate(info['medicines'], 1):
            output += f"   {i}. {med}\n"

        output += f"\n📋 DOSAGE GUIDANCE:\n   {info['dosage']}\n"
        output += f"\n⚠️ PRECAUTIONS:\n   {info['precautions']}\n"
        output += f"\n👨‍⚕️ CONSULT SPECIALIST:\n   {info['specialist']}\n"
    else:
        output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"  🔍 SEARCHING FOR: {user_query.upper()}\n"
        output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        output += "No exact match found in local symptom database.\nShowing FDA drug database results below.\n"

    if fda_results:
        output += f"\n{'='*50}\n"
        output += f"  📦 FDA DRUG DATABASE RESULTS ({len(fda_results)} found)\n"
        output += f"{'='*50}\n"
        for i, drug in enumerate(fda_results, 1):
            output += f"\n── Drug #{i} ──────────────────────────\n"
            output += f"  Brand: {drug['brand_name']}\n"
            output += f"  Generic: {drug['generic_name']}\n"
            output += f"  Indication: {drug['indication'][:250]}...\n"
            output += f"  Dosage: {drug['dosage'][:200]}...\n"
            output += f"  ⚠️ Warning: {drug['warnings'][:200]}...\n"
    elif not local_match:
        output += "\nNo FDA drug results found for this query either.\nTry common symptoms like: fever, headache, cold, cough, back pain, diabetes, etc.\n"

    output += "\n\n" + "─"*50
    output += "\n⚠️ DISCLAIMER: This is for informational purposes only.\n"
    output += "Always consult a licensed physician before taking any medication.\n"
    output += "─"*50

    return output
