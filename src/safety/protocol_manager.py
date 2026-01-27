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
        # COMPREHENSIVE VOCABULARY MAP
        # Maps visual symptoms -> Clinical Protocol Keys
        key_map = {
            "infection_suspected": [
                # Visual Cues (Colors/Textures)
                "pus", "purulent", "purulence", "yellow discharge", "green discharge",
                "cloudy exudate", "thick drainage", "biofilm",
                # Clinical Signs
                "abscess", "cellulitis", "erythema", "redness", "rubor",
                "swelling", "edema", "warmth", "hot to touch", "fluctuance",
                "foul odor", "malodor", "smell", "crepitus",
                # Diagnostic Terms
                "infection", "infected", "septic", "sepsis", "bacterial"
            ],
            
            "pressure_injury": [
                # Locations
                "sacrum", "coccyx", "heel", "ischial", "trochanter", "bony prominence",
                # Stages & Tissue Types
                "pressure ulcer", "pressure injury", "decubitus", "bedsore",
                "stage 1", "stage 2", "stage 3", "stage 4", "unstageable",
                "deep tissue injury", "dti",
                # Necrosis triggers Pressure Protocol (needs offloading/debridement)
                "eschar", "black tissue", "necrotic", "necrosis", "slough",
                "fibrin", "dead tissue"
            ],
            
            "diabetic_foot_ulcer": [
                "diabetic", "dfu", "dm ulcer", "neuropathic", "neuroischemic",
                "plantar", "metatarsal head", "charcot", "callus", "hyperkeratosis",
                "gangrene", "toes"
            ],
            
            "venous_leg_ulcer": [
                "venous", "vlu", "stasis", "varicose", "insufficiency",
                "gaiter area", "medial malleolus", "hemosiderin", "staining",
                "lipodermatosclerosis", "champagne bottle leg", "atrophie blanche"
            ],
            
            "surgical_wound": [
                "surgical", "incision", "post-op", "postop", "suture", "staples",
                "dehiscence", "evisceration", "operation site", "graft"
            ]
        }

        # PRIORITY SEARCH ORDER
        # We check Critical conditions first.
        search_order = [
            "infection_suspected", # Highest Risk (Sepsis)
            "pressure_injury",     # High Risk (Necrosis)
            "diabetic_foot_ulcer", # High Risk (Amputation)
            "surgical_wound",      # Moderate Risk
            "venous_leg_ulcer"     # Chronic
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
        
        # If no keywords matched, assume the patient is Stable/Healthy.
        return self.protocols.get("general_care", {
            "name": "General Wound Care",
            "severity": "Low",
            "management": ["Monitor for changes", "Keep area clean"]
        })
