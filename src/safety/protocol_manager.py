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
                self._load_defaults()
        else:
            logger.warning(f"⚠️ {path} not found. Loading defaults.")
            self._load_defaults()

    def get_protocol(self, analysis_text):
        """
        Scans the diagnosis text for keywords and returns the matching protocol.
        """
        if not analysis_text:
            logger.warning("ProtocolManager received empty text. Returning Default.")
            return self.protocols.get("pressure_injury", {})
        
        text = analysis_text.lower()
        
        # 1. Map keywords to protocol keys (matches your YAML structure)
        key_map = {
            "diabetic": "diabetic_foot_ulcer",
            "dfu": "diabetic_foot_ulcer",
            "neuropathic": "diabetic_foot_ulcer",
            
            "venous": "venous_leg_ulcer",
            "vlu": "venous_leg_ulcer",
            "stasis": "venous_leg_ulcer",
            "edema": "venous_leg_ulcer",
            
            "pressure": "pressure_injury",
            "decubitus": "pressure_injury",
            "sacrum": "pressure_injury",
            "coccyx": "pressure_injury",
            
            "surgical": "surgical_wound",
            "incision": "surgical_wound",
            
            "normal": "normal_skin",
            "intact": "normal_skin"
        }

        # 2. Find match
        matched_protocol = None
        for keyword, protocol_key in key_map.items():
            if keyword in text:
                matched_protocol = self.protocols.get(protocol_key)
                break
        
        # 3. Format Output
        if matched_protocol:
            return matched_protocol
        else:
            # SAFETY CATCH: If text mentions "Necrosis" or "Eschar" but no specific condition,
            # default to Pressure Injury protocol as it covers debridement best.
            if "necrosis" in text or "eschar" in text or "black" in text:
                return self.protocols.get("pressure_injury", {
                    "name": "Necrotic Wound (Unspecified)",
                    "management": ["Debridement indicated", "Consult Wound Specialist"]
                })
            
            return {
                "name": "Unspecified Condition",
                "severity": "Unknown",
                "management": ["Requires professional clinical evaluation.", "Protocol not found for this classification."]
            }
