import pandas as pd
import re

def normalize_query(query):
    """
    Quick preprocessing for common issues.
    Handles typos, punctuation, and case normalization.
    """
    query = query.lower().strip()
    
    # Fix common typos found in testing
    query = query.replace("critcal", "critical")
    query = query.replace("urgnet", "urgent")
    query = query.replace("pateint", "patient")
    query = query.replace("patinet", "patient")
    
    # Remove trailing punctuation (e.g. "critical?")
    query = re.sub(r'[?!.]+$', '', query)
    
    return query

def get_holistic_answer(query, df):
    """
    Enhanced Rule-Based Logic with Intent Classification & Negation Handling.
    """
    # 1. Preprocessing
    query = normalize_query(query)
    
    # Debug print (Check your terminal!)
    print(f"\n[DEBUG] Processing Query: '{query}'")
    
    # 2. Strict Negation Check
    negation_tokens = {'not', 'no', "isn't", "aren't", 'except', 'non'}
    query_tokens = set(query.split())
    has_negation = not query_tokens.isdisjoint(negation_tokens)

    # 3. Define Keywords
    severity_keywords = ["critical", "urgent", "priority", "high risk", "severe", "danger", "deteriorating"]
    count_keywords = ["how many", "count", "number", "total", "show", "list", "find"]
    patient_keywords = ["patient", "case", "person", "people"]
    
    # 4. Check Intents
    has_severity = any(k in query for k in severity_keywords)
    has_count = any(k in query for k in count_keywords)
    has_patient = any(k in query for k in patient_keywords)
    
    # Specific Disease Checks
    has_diabetes = "diabet" in query
    has_hypertension = "hyperten" in query
    has_comorbidity = "comorbidit" in query

    # --- RULE 0: TRIAGE / CRITICAL (Top Priority) ---
    if has_severity and (has_count or has_patient):
        report = generate_priority_report(df)
        
        if has_negation:
            # "How many NOT critical"
            total = len(df['Patient_ID'].unique())
            critical = len([x for x in report if x['Severity'] == 'Critical'])
            urgent = len([x for x in report if x['Severity'] == 'Urgent'])
            stable = total - critical - urgent
            return (
                f"**📊 Non-Critical Patient Status:**\n\n"
                f"Out of {total} total patients:\n"
                f"- **✅ Stable/Low Risk:** {stable} patients ({(stable/total*100):.1f}%)\n\n"
                f"*(Excludes {critical} critical + {urgent} urgent cases)*"
            )
        else:
            # "How many critical"
            critical_count = len([x for x in report if x['Severity'] == 'Critical'])
            urgent_count = len([x for x in report if x['Severity'] == 'Urgent'])
            total_patients = len(df['Patient_ID'].unique())
            
            response = f"**🚨 Patient Risk Status:**\n\n"
            if critical_count > 0: response += f"- **🔴 Critical:** {critical_count} patient{'s' if critical_count!=1 else ''}\n"
            if urgent_count > 0: response += f"- **⚠️ Urgent:** {urgent_count} patient{'s' if urgent_count!=1 else ''}\n"
            stable_count = total_patients - critical_count - urgent_count
            response += f"- **✅ Stable:** {stable_count}\n\n*View details in Dashboard.*"
            return response

    # --- RULE 1: SPECIFIC DISEASES (Top Priority) ---
    # Triggered by "Diabetes", "Hypertension", or "Comorbidity"
    if has_diabetes or has_hypertension or has_comorbidity:
        if 'Comorbidities' in df.columns:
            unique_patients = df.drop_duplicates(subset=['Patient_ID'])
            total = len(unique_patients)
            
            # Convert to string first to prevent "Can only use .str accessor" error
            comor_series = unique_patients['Comorbidities'].fillna('').astype(str)
            
            diabetes = comor_series.str.contains(r'Diabet|DM|T2DM', case=False, regex=True).sum()
            hyper = comor_series.str.contains(r'Hyperten|HTN|HBP', case=False, regex=True).sum()
            
            return (
                f"**📊 Comorbidity Snapshot:**\n"
                f"Out of {total} patients:\n"
                f"- **Diabetes:** {diabetes} patients\n"
                f"- **Hypertension:** {hyper} patients\n"
                f"*(Based on keyword search in records)*"
            )

    # --- RULE 1.5: AGE FILTERING (Fix for "Age > 50" queries) ---
    if "age" in query and ('patient' in query or 'people' in query):
        # Regex to find "more than 50", "> 50", "over 65", etc.
        match = re.search(r'(more than|greater than|over|above|>\s*)\s*(\d+)', query)
        if match:
            threshold = int(match.group(2))
            if 'Age' in df.columns:
                # Convert Age to numeric, coercing errors to NaN
                df['Age_Num'] = pd.to_numeric(df['Age'], errors='coerce')
                count = len(df[df['Age_Num'] > threshold])
                return (
                    f"**📊 Age Demographics:**\n\n"
                    f"There are **{count} patients** with age > {threshold}.\n"
                    f"*(Source: Direct query of patient records)*"
                )

    # --- RULE 2: GENDER DISTRIBUTION ---
    if "gender" in query or "sex" in query or "male" in query or "female" in query:
        if 'Sex' in df.columns:
            unique_patients = df.drop_duplicates(subset=['Patient_ID'])
            dist = unique_patients['Sex'].value_counts()
            
            # Full Detail Version
            report = "**📊 Gender Distribution:**\n"
            for sex, count in dist.items():
                pct = (count / len(unique_patients)) * 100
                report += f"- **{sex}:** {count} ({pct:.1f}%)\n"
            return report

    # --- RULE 3: WOUND STATISTICS ---
    if "wound" in query and ("avg" in query or "average" in query or "size" in query or "stat" in query):
        if 'Wound_Size_Length_cm' in df.columns and 'Wound_Size_Width_cm' in df.columns:
            # Full Detail Version
            avg_l = df['Wound_Size_Length_cm'].mean()
            avg_w = df['Wound_Size_Width_cm'].mean()
            max_l = df['Wound_Size_Length_cm'].max()
            
            return (
                f"**📊 Wound Statistics (All Visits):**\n"
                f"- **Average Size:** {avg_l:.1f}cm x {avg_w:.1f}cm\n"
                f"- **Largest Recorded Length:** {max_l}cm"
            )

    # --- RULE 4: TOTAL PATIENT COUNT (Lowest Priority) ---
    # Moved to bottom so it catches only what falls through
    if has_count and has_patient:
        if 'Patient_ID' in df.columns:
            count = df['Patient_ID'].nunique()
            ids = sorted(df['Patient_ID'].unique())
            
            # Full Detail Version (ID Preview)
            id_list_str = [str(x) for x in ids[:5]]
            id_preview = ", ".join(id_list_str) + ("..." if len(ids) > 5 else "")
            
            return (
                f"**📊 Holistic Database Summary:**\n\n"
                f"There are a total of **{count} unique patients** in the database.\n"
                f"**Patient IDs:** {id_preview}\n\n"
                f"*(Source: Direct calculation from Patient Records)*"
            )

    return None