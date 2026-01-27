# build_release.py
from setuptools import setup
from Cython.Build import cythonize
import os

# 1. Target the UNIFIED file
files_to_compile = [
    "src/engine/engine_core.py" 
]

# 2. Build
setup(
    ext_modules=cythonize(
        files_to_compile,
        compiler_directives={'language_level': "3", 'always_allow_keywords': True},
        build_dir="build_temp"
    )
)

# 3. Rename to engine_core.so (The Smart Fix)
print("\n🧹 Finalizing Binary Name...")
base_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_path, "src", "engine")

for file in os.listdir(src_path):
    if file.startswith("engine_core") and file.endswith(".so"):
        old_path = os.path.join(src_path, file)
        # We rename it to engine_core.so to match the Python file name
        new_path = os.path.join(src_path, "engine_core.so") 
        
        if os.path.exists(new_path):
            os.remove(new_path)
            
        os.rename(old_path, new_path)
        print(f"✅ Created Binary: {new_path}")

print("\n🚀 Build Complete.")