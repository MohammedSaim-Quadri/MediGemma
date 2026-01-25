# build_release.py
from setuptools import setup
from Cython.Build import cythonize

# THE CORE IP LIST (Secret Sauce)
files_to_compile = [
    "src/engine/rag.py",                # Prompts & Logic
    "src/engine/vision.py",             # Vision Pipeline
    "src/core/router.py",               # Intent Router
    "src/core/priority_rules.py",       # Triage Math
    "src/safety/protocol_manager.py",   # Protocols
    "src/safety/verifier.py"            # Guardrails
]

setup(
    ext_modules=cythonize(
        files_to_compile,
        compiler_directives={
            'language_level': "3",
            'always_allow_keywords': True
        },
        build_dir="build_temp"
    )
)