import yaml
import logging
import os

logger = logging.getLogger(__name__)

class ProtocolManager:
    """
    The 'Safety Guardrail' that maps diagnoses to Evidence-Based Protocols.
    Reads from config/protocols.yaml so rules are easy to audit.
    """
    def __init__(self, protocols_path="config/protocols.yaml"):
        self.protocols = {}
        self._load_protocols(protocols_path)
        self._load_defaults()

    def _load_protocols(self, path):
        """Loads clinical protocols from an external YAML file."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    # Support both structure formats (direct dict or under 'clinical_protocols')
                    self.protocols = data.get('clinical_protocols', data)
                logger.info(f"✅ Loaded {len(self.protocols)} protocols from {path}")
            except Exception as e:
                logger.error(f"❌ Failed to load protocols.yaml: {e}")
        else:
            logger.warning(f"⚠️ {path} not found. Loading defaults.")

    def _load_defaults(self):
        # Fallback protocols if YAML is missing specific keys
        defaults = {
            "infection_control": {
                "name": "Active Infection Protocol",
                "severity": "High",
                "management": ["Obtain Culture", "Start Empiric Antibiotics", "Surgical Consult"]
            },
            "diabetic_foot_ulcer": {
                "name": "Diabetic Foot Ulcer", 
                "severity": "Medium",
                "management": ["Offloading", "Glycemic Control", "Debridement"]
            }
        }
        # Merge defaults only if key doesn't exist
        for k, v in defaults.items():
            if k not in self.protocols:
                self.protocols[k] = v
    
    def get_protocol(self, analysis_text):
        """
        Scans the diagnosis text for keywords and returns the matching protocol.
        """
        if not analysis_text:
            logger.warning("ProtocolManager received empty text. Returning Default.")
            return self.protocols.get("infection_suspected", {})
        
        # Convert to string safely
        text = str(analysis_text).lower()
        
        # 1. Map keywords to protocol keys
        key_map = {
            "infection_suspected": ["pus", "purulent", "odor", "abscess", "cellulitis", "foul", "erythema", "warmth", "fluctuance"],
            "diabetic_foot_ulcer": ["diabetic", "dfu", "neuropathic", "blood sugar", "neuroischemic", "charcot"],
            "pressure_injury": ["pressure", "decubitus", "sacrum", "coccyx", "heel", "stage", "bedsore"],
            "venous_leg_ulcer": ["venous", "vlu", "stasis", "varicose", "hemosiderin"],
            "surgical_wound": ["surgical", "incision", "suture", "dehiscence", "operation", "staples", "postop"],
            
            # The "Visual Bridges" (Critical for Vision Models)
            "necrotic_tissue": ["necrotic", "necrosis", "eschar", "black", "slough", "yellow", "dead tissue", "brown"],
            "wound_drainage": ["weepy", "exudate", "drainage", "oozing", "wet", "moist", "serous", "discharge"],
            
            # Lowest Priority
            "normal_skin": ["normal", "intact", "healthy", "no wound"]
        }

        search_order = [
            "infection_suspected",  # 1. Life Threatening
            "necrotic_tissue",      # 2. Tissue Threatening (Gangrene/Slough)
            "diabetic_foot_ulcer",  # 3. High Risk Etiology
            "pressure_injury",      # 4. High Risk Etiology
            "surgical_wound",       # 5. Acute Risk
            "venous_leg_ulcer",     # 6. Chronic Risk
            "wound_drainage",       # 7. Active Symptom (The "Weepy" catch)
            "normal_skin"           # 8. Nothing wrong found
        ]

        matched_protocol_key = None

        for protocol_key in search_order:
            keywords = key_map.get(protocol_key, [])
            # specific check: if word is in text
            if any(k in text for k in keywords):
                matched_protocol_key = protocol_key
                break # STOP looking. We found the most severe issue.

        # --- Return the Match ---
        if matched_protocol_key:
            return self.protocols.get(matched_protocol_key, {
                "name": f"Protocol: {matched_protocol_key.replace('_', ' ').title()}", 
                "management": ["Refer to standard clinical guidelines."]
            })
        
        # Fallback if AI output was total gibberish (e.g., "The image is blurry")
        return {
            "name": "Unclassified Wound Findings",
            "severity": "Unknown",
            "management": ["Manual clinical assessment required.", "Verify image quality."]
        }
